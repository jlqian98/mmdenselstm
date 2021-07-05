from torch.utils import data
from torch.utils.data import DataLoader
import torch
from librosa.util import find_files
import numpy as np
import os
from hparams import create_hparams

import pandas as pd


class MUSDB(data.Dataset):

    def __init__(self, file_path, out_channels_list):
        self.file_list = find_files(file_path, 'npz')
        self.out_channels_list = out_channels_list


    def __len__(self, ):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        spec = data['data']
        label = data['label']
        label = label[self.out_channels_list]

        return torch.from_numpy(spec).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)


def get_loader(hparams):

    train_path = os.path.join(hparams.fft_path, 'train')
    eval_path = os.path.join(hparams.fft_path, 'valid')
    train = MUSDB(train_path, hparams.out_channels_list)
    valid = MUSDB(eval_path, hparams.out_channels_list)

    valid_loader = DataLoader(dataset=valid,
                              batch_size=hparams.batch_size,
                              shuffle=True,
                              num_workers=hparams.num_workers)

    train_loader = DataLoader(dataset=train,
                              batch_size=hparams.batch_size,
                              shuffle=True,
                              num_workers=hparams.num_workers)

    return train_loader, valid_loader
