import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import os
import argparse
from hparams import create_hparams

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import MUSDB, get_loader
from trainer import Trainer


def train(args):
    hparams = create_hparams()
    hparams.warm_start = args.warm_start
    hparams.checkpoint = args.c

    train_loader, val_loader = get_loader(hparams)
    trainer = Trainer(hparams, train_loader, val_loader)
    trainer.train()

if __name__ == '__main__':

    # python train.py --warm_start -c checkpoints/unet/epoch_340.pkl
    parser = argparse.ArgumentParser()
    parser.add_argument("--warm_start", action='store_true', help="whether use pretrained model")
    parser.add_argument("-c", default=None, type=str, help="the pretrained model path")
    args = parser.parse_args()
    train(args)
