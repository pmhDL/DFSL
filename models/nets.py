import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.misc import euclidean_metric, cosine_metric, cosloss
from torch.nn.utils.weight_norm import WeightNorm
from torch.distributions import Normal, kl_divergence
from models.res12 import Res12
from models.wrn28 import Wrn28


def compute_class_dstr(mu, logvar, lb, args):
    var = torch.exp(logvar)
    M = []
    C = []
    if args.merge_dstr == 'mean':
        for y in torch.unique(lb):
            id = torch.where(lb == y)[0]
            mu_ = mu[id].mean(0)
            var_ = var[id].mean(0)
            M.append(mu_)
            C.append(var_)
    elif args.merge_dstr == 'merge':
        for y in torch.unique(lb):
            id = torch.where(lb == y)[0]
            mu_ = mu[id]
            var_ = var[id]
            Cmean = ((var_ ** (-1)).sum(0)) ** (-1) # Csum = ((var_ ** (-1)).mean(0)) ** (-1)
            Mnew = Cmean * (((var_ ** (-1)) * mu_).sum(0))
            Csum = ((var_ ** (-1)).mean(0)) ** (-1)
            M.append(Mnew)
            C.append(Csum)
    M = torch.stack(M, 0)
    C = torch.stack(C, 0)
    return M, C


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        WeightNorm.apply(self.L, 'weight', dim=0)
        if outdim <= 200:
            self.scale_factor = 2
        else:
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)
        return scores


class Model(nn.Module):
    def __init__(self, args, mode='pre', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        if self.args.dataset == 'cub':
            sdim = 312
        else:
            sdim = 300
        if self.args.model_type == 'res12':
            self.encoder = Res12()
            z_dim = 640
        elif self.args.model_type == 'wrn28':
            self.encoder = Wrn28()
            z_dim = 640
        self.z_dim = z_dim
        self.sdim = sdim
        self.latentdim = 512

        if self.mode == 'pre':
            self.pre_fc = distLinear(z_dim, num_cls)
            self.rot_fc = nn.Linear(z_dim, 4)
        elif self.mode == 'initde' or self.mode == 'dfsl':
            self.mseloss = torch.nn.MSELoss()
            self.classifer = distLinear(z_dim, num_cls)
            self.Nete = nn.Linear(z_dim, sdim)
            self.Netd = nn.Linear(sdim, z_dim)
            self.RecNet = nn.Linear(self.latentdim, sdim + z_dim)
            self.MuNet = nn.Linear(sdim + z_dim, self.latentdim)
            self.VarNet = nn.Linear(sdim + z_dim, self.latentdim)


    def forward(self, inp):
        if self.mode == 'pre':
            return self.forward_pre(inp)
        elif self.mode == 'preval':
            datas, dataq = inp
            return self.forward_preval(datas, dataq)
        elif self.mode == 'initde':
            data, datac, sem = inp
            return self.forward_initde(data, datac, sem)
        elif self.mode == 'dfsl':
            datas, ys, sem, dataq = inp
            return self.forward_dfsl(datas, ys, sem, dataq)
        else:
            raise ValueError('Please set the correct mode.')


    def forward_pre(self, inp):
        embedding = self.encoder(inp)
        logits = self.pre_fc(embedding)
        rot = self.rot_fc(embedding)
        return logits, rot


    def forward_preval(self, data_shot, data_query):
        proto = self.encoder(data_shot)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)
        query = self.encoder(data_query)
        if self.args.metric == 'ED':
            logitq = euclidean_metric(query, proto)
        elif self.args.metric == 'cos':
            x_mul = torch.matmul(query, proto.T)
            Normv = torch.mul(torch.norm(query, dim=1).unsqueeze(1), torch.norm(proto, dim=1).unsqueeze(0))
            logitq = torch.div(x_mul, Normv)
        return logitq


    def forward_initde(self, data, datac, sem):
        sem_m = self.Nete(data)
        losslatent = self.mseloss(sem_m, sem) #losslatent = cosloss(sem_m, sem)
        recX = self.Netd(sem_m)
        lossrec = self.mseloss(recX, datac) #lossrec = cosloss(recX, datac)
        logit = self.classifer(recX)
        return logit, losslatent, lossrec


    def forward_dfsl(self, datas, ys, sem, dataq):
        sz = datas.size(0)
        proto = F.normalize(datas, 1)
        query = F.normalize(dataq, 1)
        proto = proto.reshape(self.args.shot, self.args.way, -1).mean(dim=0)

        if self.args.metric == 'ED':
            logit0 = euclidean_metric(query, proto)
        elif self.args.metric == 'cos':
            x_mul = torch.matmul(query, proto.T)
            Normv = torch.mul(torch.norm(query, dim=1).unsqueeze(1), torch.norm(proto, dim=1).unsqueeze(0))
            logit0 = torch.div(x_mul, Normv)

        # --------- autoencoder debiasing ----------
        optimizer1 = torch.optim.Adam([
            {'params': self.Nete.parameters()},
            {'params': self.Netd.parameters()}
        ], lr=self.args.lr, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=50, gamma=0.2)
        self.Nete.train()
        self.Netd.train()
        for i in range(30):
            laten = self.Nete(datas)
            losslatent = self.mseloss(laten, sem)
            rec = self.Netd(laten)
            lossrec = self.mseloss(rec, datas)
            loss = losslatent + lossrec
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            lr_scheduler.step()
        self.Nete.eval()
        self.Netd.eval()

        data = torch.cat([datas, dataq], dim=0)
        data = self.Nete(data)
        semq = data[sz:].detach()
        data = self.Netd(data)
        datas = data[:sz].detach()
        dataq = data[sz:].detach()

        # --------- train p2d transformer ----------
        inpS = torch.cat([datas, sem], dim=1)
        inpQ = torch.cat([dataq, semq], dim=1)

        optimizer2 = torch.optim.Adam([
            {'params': self.MuNet.parameters()},
            {'params': self.VarNet.parameters()},
            {'params': self.RecNet.parameters()}
        ], lr=self.args.lr, weight_decay=1e-5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=50, gamma=0.2)
        self.MuNet.train()
        self.VarNet.train()
        self.RecNet.train()
        for i in range(30):
            mu = self.MuNet(inpS)
            logvar = self.VarNet(inpS)
            std = torch.exp(0.5 * logvar)

            dtr1 = torch.distributions.normal.Normal(mu, std)
            I = torch.ones_like(mu).type(mu.type())
            Z = torch.zeros_like(mu).type(mu.type())
            dtr2 = torch.distributions.normal.Normal(Z, I)
            losskl = torch.distributions.kl.kl_divergence(dtr1, dtr2).sum(-1).mean()

            latentx = dtr1.rsample()
            recX = self.RecNet(latentx)
            p_x = torch.distributions.Normal(recX, torch.ones_like(recX))
            losslog = p_x.log_prob(inpS).sum(-1).mean()

            loss = losskl - losslog
            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()
            lr_scheduler.step()
        self.MuNet.eval()
        self.VarNet.eval()
        self.RecNet.eval()

        mus = self.MuNet(inpS)
        logvars = self.VarNet(inpS)
        mus, var = compute_class_dstr(mus, logvars, ys, self.args)
        stds = var ** 0.5

        muq = self.MuNet(inpQ)
        logvarq = self.VarNet(inpQ)
        stdq = torch.exp(0.5 * logvarq)
        
        Logit = []
        for k in range(muq.size(0)):
            dbt1 = Normal(muq[k].view(1, -1), stdq[k].view(1, -1))
            dbt2 = Normal(mus, stds)
            sim = -kl_divergence(dbt1, dbt2).mean(-1)
            Logit.append(sim)
        logit1 = torch.stack(Logit, dim=0)
        return logit1