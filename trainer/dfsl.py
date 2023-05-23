import os.path as osp
import os
import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader.samplers import CategoriesSampler
from models.nets import Model
from utils.misc import Averager, count_acc, ensure_path, compute_confidence_interval
from tensorboardX import SummaryWriter
from dataloader.dataset_loaderf import DatasetLoader as Dataset
import numpy as np


class DFSLTrainer(object):
    def __init__(self, args):
        # Set the folder to save the records and checkpoints
        log_base_dir = './logs/'
        if not osp.exists(log_base_dir):
            os.mkdir(log_base_dir)
        pre_base_dir = osp.join(log_base_dir, 'dfsl')
        if not osp.exists(pre_base_dir):
            os.mkdir(pre_base_dir)
        save_path1 = '_'.join([args.dataset, args.model_type, 'addv=' + str(args.addVal)])
        args.save_path = pre_base_dir + '/' + save_path1
        ensure_path(args.save_path)

        self.args = args
        # Load pretrain set
        self.trainset = Dataset('val', self.args)
        self.train_loader = DataLoader(dataset=self.trainset, batch_size=args.pre_batch_size, shuffle=True,
                                       num_workers=8, pin_memory=True)
        # Set pretrain class number
        num_class_pretrain = self.trainset.num_class
        # Build pretrain model
        self.model = Model(self.args, mode='initde', num_cls=num_class_pretrain)

        if self.args.opt == 'SGD':
            self.optimizer = torch.optim.SGD([
                 {'params': self.model.Nete.parameters()},
                 {'params': self.model.Netd.parameters()},
                 {'params': self.model.classifer.parameters()},
                ], lr=self.args.lr)
        elif self.args.opt == 'Adam':
            self.optimizer = torch.optim.Adam([
                        {'params': self.model.Nete.parameters()},
                        {'params': self.model.Netd.parameters()},
                        {'params': self.model.classifer.parameters()},
                        ], lr=self.args.lr, weight_decay=self.args.pre_weight_decay)
        # Set learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.scheduler_milestones,
                                                                 gamma=args.pre_gamma)
        # Set model to GPU
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.model = self.model.cuda()


    def save_model(self, name):
        """The function to save checkpoints.
        Args:
          name: the name for saved checkpoint
        """
        torch.save(dict(params=self.model.state_dict()), osp.join(self.args.save_path, name + '.pth'))


    def train(self):
        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['train_loss'] = []
        trlog['val_loss'] = []
        trlog['train_acc'] = []
        trlog['val_acc'] = []
        trlog['max_acc'] = 0.0
        trlog['max_acc_epoch'] = 0
        # Set global count to zero
        global_count = 0
        # Set tensorboardX
        writer = SummaryWriter(comment=self.args.save_path)
        # Start pretrain
        for epoch in range(1, self.args.max_epoch + 1):
            # Update learning rate
            self.lr_scheduler.step()
            # Set the model to train mode
            self.model.train()
            self.model.mode = 'initde'
            # Set averager classes to record training losses and accuracies
            train_loss_averager = Averager()
            train_acc_averager = Averager()
            # Using tqdm to read samples from train loader
            tqdm_gen = tqdm.tqdm(self.train_loader)
            for i, batch in enumerate(tqdm_gen, 1):
                self.optimizer.zero_grad()
                # Update global count number
                global_count = global_count + 1
                if torch.cuda.is_available():
                    data, datac, label, sem = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    datac = batch[1]
                    label = batch[2]
                    sem = batch[3]
                sem = sem.type(data.type())
                # Output logits for model
                logit, losslatent, lossrecon = self.model((data, datac, sem))
                # Calculate train loss
                loss_cls = F.cross_entropy(logit, label)
                loss = loss_cls + losslatent + lossrecon
                # Calculate train accuracy
                acc = count_acc(logit, label)
                # Write the tensorboardX records
                writer.add_scalar('data/loss', float(loss), global_count)
                writer.add_scalar('data/acc', float(acc), global_count)
                # Print loss and accuracy for this step
                tqdm_gen.set_description('Epoch {}, Loss={:.4f} Acc={:.4f}'.format(epoch, loss.item(), acc))
                # Add loss and accuracy for the averagers
                train_loss_averager.add(loss.item())
                train_acc_averager.add(acc)
                # Loss backwards and optimizer updates
                loss.backward()
                self.optimizer.step()
            # Save model every 10 epochs
            if epoch % 10 == 0:
                self.save_model('max_acc')
        writer.close()

    def eval(self):
        """ test phase."""
        # Load test set
        test_set = Dataset('test', self.args)
        sampler = CategoriesSampler(test_set.label, 600, self.args.way, self.args.shot + self.args.val_query)
        loader = DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
        # Set test accuracy recorder
        test_acc_record = np.zeros((600,))
        self.model.load_state_dict(torch.load(osp.join(self.args.save_path, 'max_acc' + '.pth'))['params'])

        # Set model to eval mode
        self.model.eval()
        self.model.mode = 'dfsl'
        # Set accuracy averager
        ave_acc1 = Averager()

        # Generate labels
        label = torch.arange(self.args.way).repeat(self.args.val_query)
        label_shot = torch.arange(self.args.way).repeat(self.args.shot)
        if torch.cuda.is_available():
            label = label.type(torch.cuda.LongTensor)
            label_shot = label_shot.type(torch.cuda.LongTensor)
        else:
            label_shot = label_shot.type(torch.LongTensor)
            label = label.type(torch.LongTensor)

        for i, batch in enumerate(loader, 1):
            if torch.cuda.is_available():
                if torch.cuda.is_available():
                    data, _, _, sem = [_.cuda() for _ in batch]
                else:
                    data = batch[0]
                    sem = batch[3]
            p = self.args.shot * self.args.way
            datas, dataq = data[:p], data[p:]
            sem = torch.tensor(sem).type(datas.type())
            sems = sem[:p]

            logit1 = self.model((datas, label_shot, sems, dataq))
            acc1 = count_acc(logit1, label)
            ave_acc1.add(acc1)

            test_acc_record[i - 1] = acc1
            if i % 100 == 0:
                print('batch {}: {:.2f} '.format(i, ave_acc1.item() * 100))

        # Calculate the confidence interval, update the logs
        m, pm = compute_confidence_interval(test_acc_record)
        print('Test Acc {:.4f} + {:.4f}'.format(m, pm))