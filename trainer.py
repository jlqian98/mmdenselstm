import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np
import time

from torch.autograd import Variable
from torch import optim
from models.model import Model
import torch.nn.functional as F
from utils import spec_loss, save_model, load_gt, EarlyStopping, load_mix_gt

from numpy.random import normal

from logger import Logger
from tqdm import tqdm
import librosa
import argparse
import gc


class Trainer(object):

    def __init__(self, hparams, full_loader, val_loader):
        """
        Init the trainer
        """
        self.hparams = hparams
        
        self.model = Model(self.hparams)
        # dataset
        self.full_loader = full_loader
        self.val_loader = val_loader

        # training setup
        self.recon_criterion = nn.MSELoss()
        self.batch_size = hparams.batch_size
        self.save_iter = hparams.save_iter
        self.logger = self.model.logger
        self.device = hparams.device
        self.num_epoch = hparams.num_epoch
        self.sdr_iter = hparams.sdr_iter

        self.test_dir = os.path.join(hparams.wave_path, 'test')  
        self.test_files = os.listdir(self.test_dir)

        self.earlystopping = EarlyStopping(hparams.patience)

        
    def eval_metric(self, epoch, ):
        """
        evaluate the sdr metric for the model
        return: (sdr, sir, sar)
        """
        # test_idx = np.random.randint(50)
        print("eval sdr...")
        test_idx = 9
        test_dir = os.path.join(self.test_dir, self.test_files[test_idx])
        mix, gt = load_gt(self.hparams, test_dir, duration=60)
        sdr, sir, sar = self.model.evaluate(mix, gt)
        print(sdr, sir, sar)
        self.logger.log("evaluation.metrics.sdr", sdr.mean(), epoch)


    # TODO log training status
    def evaluate(self, epoch, data_loader, criterion):
        """
        evaluate the validation loss of the model (including earlystopping)
        
        """
        self.model.net.eval()
        loss_total = 0.0
        with torch.no_grad():
            for idx, (mix, sep) in enumerate(data_loader):
                mix = mix.to(self.hparams.device)
                sep = sep.to(self.hparams.device)
                # mask = self.model.net(mix)
                # predict = mix * mask
                predict = self.model.net(mix)
                loss = spec_loss(predict, sep, criterion)
                loss_total += loss.item()
        loss_eval = loss_total / len(data_loader)
        print("Evaluation epoch: %5d loss: %.10f]" %
              (epoch, loss_eval))
        self.logger.log("evaluation.loss", loss_eval, epoch)
        self.earlystopping(loss_eval, self.model)

        # check
        # if epoch % self.save_iter == 0:
        file_dir = os.path.join(self.hparams.wave_path, "test", "Al James - Schoolboy Facination")
        mix, gt = load_gt(self.hparams, file_dir, 15)
        self.model.spec_plot(mix, gt, epoch)


    def train_epoch(self, epoch, data_loader, criterion):
        start = time.time()
        self.model.net.train()
        loss_total = 0.0
        for idx, (mix, sep) in enumerate(data_loader):
            mix = mix.to(self.hparams.device)
            sep = sep.to(self.hparams.device)

            # mask = self.model.net(mix)
            # predict = mix * mask
            predict = self.model.net(mix)
            loss = spec_loss(predict, sep, criterion)
            
            loss_total += loss.item()
            loss.backward()

            self.model.optim.step()
            self.model.reset_grad()
            self.logger.log("training.loss", loss.item(), epoch * len(data_loader) + idx)

            print("\r" + "epoch: [%d/%d]  iter: [%d/%d]  training loss: %.8f time: %.4fs" % (epoch, self.hparams.num_epoch, idx+1, len(data_loader), loss_total / (idx+1), time.time()-start), end='')

        self.model.scheduler.step()
        print()
            

    def train(self):
        """
        training the model
        """
        while self.model.epoch <= self.num_epoch:
            self.evaluate(self.model.epoch, self.val_loader, self.recon_criterion)
            self.train_epoch(self.model.epoch, self.full_loader, self.recon_criterion)
            if self.model.epoch % self.save_iter == 0:
                self.model.save("epoch_%d" % self.model.epoch)
            self.model.save("last" % self.model.epoch)
            self.model.epoch += 1
            if self.earlystopping.early_stop:
                print('Earlystopping! Finish training!')
                break
            gc.collect()

        print("Congratulations! Finish training!")