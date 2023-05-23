import os.path as osp
from torch.utils.data import Dataset
import numpy as np

class DatasetLoader(Dataset):
    def __init__(self, setname, args):
        THE_PATH = osp.join(args.dataset_dir, 'feat-' + setname + '.npz')
        data0 = np.load(THE_PATH)
        featn = data0['features']
        label = data0['targets']

        THE_PATH_clean = osp.join(args.dataset_dir, 'feat-' + setname + '1.npz') #the unbiased features of validation set
        data1 = np.load(THE_PATH_clean)
        featc = data1['features']

        path_sem = osp.join(args.dataset_dir.split(args.dataset)[0] + args.dataset, 'few-shot-wordemb-' + setname + '.npz')
        sem = np.load(path_sem)['features']

        self.featn = featn
        self.featc = featc
        self.sem = sem
        self.label = label
        self.num_class = len(np.unique(label))

    def __len__(self):
        return len(self.featn)

    def __getitem__(self, i):
        fn = self.featn[i]
        fc = self.featc[i]
        label = self.label[i]
        sem = self.sem[label]
        return fn, fc, label, sem