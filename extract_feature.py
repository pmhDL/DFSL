import torch
from models.res12 import Res12
from models.wrn28 import Wrn28
from dataloader.dataset_loaderIm import DatasetLoader
import os
import numpy as np

def extract_feature(data_loader, setname, model, savepath, addv):
    model.eval()
    feat = []
    Lb = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = outputs.cpu().data.numpy()
            labels = labels.cpu().data.numpy()
            feat.append(outputs)
            Lb.extend(labels)
        feat = np.concatenate(feat, axis=0)
        Lb = np.array(Lb)
    print('feat shape: ', feat.shape)
    print('Lb shape: ', Lb.shape)
    if addv == '1' and setname == 'val':
        np.savez(savepath + '/feat-' + setname + '1.npz', features=feat, targets=Lb)
    else:
        np.savez(savepath + '/feat-' + setname + '.npz', features=feat, targets=Lb)
    return 0

'''------------------------params---------------------------'''
dataname = 'mini'    # mini, tiered, cub, cifar_fs
modeltype = 'res12'  # wrn28 res12
addv = '1'
datadir = '/data/FSL/dataIm/'+dataname
checkpointpath = '/data/FSL/checkpoints/'+dataname+'_'+modeltype+'_addv='+addv+'/max_acc.pth'
savepath = '/data/FSL/features/' + dataname + '/' + modeltype

cuda_device = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device
print('Using gpu:', cuda_device)

'''----------------------- build model ----------------------------'''
if modeltype == 'res12':
    model = Res12()
elif modeltype == 'wrn28':
    model = Wrn28()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    model = model.cuda()
# model = torch.nn.DataParallel(model).cuda()

'''------------------------ load checkpoints -----------------------'''
model_dict = model.state_dict()
print('model_dict: ', model_dict.keys())
print('======================================')
state = torch.load(checkpointpath)['params']
state = {k.replace('module.', ''): v for k, v in state.items()}
state = {k: v for k, v in state.items() if k in model_dict}
print('state: ', state.keys())
model_dict.update(state)
model.load_state_dict(model_dict)

'''------------------------ extract features ------------------------'''
setname = 'val'
dataset = DatasetLoader(setname, datadir, modeltype)
data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=128,
                                          shuffle=False, num_workers=12, pin_memory=True)

extract_feature(data_loader, setname, model, savepath, addv)