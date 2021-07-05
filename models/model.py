import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from utils import to_var
import torch.optim as optim
import os
from utils import wave2spec, spec2wave, align
from logger import Logger
from mir_eval.separation import bss_eval_sources

from models.mmdensenet import MMDenseNet
from models.mdensenet import MDenseNet
from models.mdenselstm import MDenseLSTM
from models.mmdenselstm import MMDenseLSTM


def separate(model, mix, args):
    """
    - model: model to generate mask
    - mix: input mix (1, T)
    return: [drums, bass, other, vocals]
    """
    model.eval()
    spec, phase, tmplen = wave2spec(args, mix)      # mix spec (1, 1, F, T)

    freq, leng = spec.shape[0], spec.shape[1]
    tmp = np.zeros((freq, align(leng)), dtype=np.float32)
    spec = np.concatenate((spec, tmp), axis=1)

    spec = torch.FloatTensor(spec[None, None, :, :])
    model, spec = model.cpu(), spec.cpu()

    inst_specs = model(spec)  # (1, 4, 512, T)

    inst_specs = inst_specs[0, :, :, :leng].detach().numpy()   # (4, 512, T)

    inst_specs = tuple([inst_specs[index, ...] for index in range(len(args.insts))])
    out = [spec2wave(args, inst_spec, phase[:, :leng], tmplen) for inst_spec in inst_specs]

    return out


def eval(ref, est):
    """ 
    ref: np.array (4, ...)
    est: np.array (4, ...)
    """
    sdr, sir, sar, _ = bss_eval_sources(ref, est)
    return sdr, sir, sar


def xavier_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0.1)


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


class Model(object):
    def __init__(self, args):
        self.args = args
        if args.model_name == 'mmdensenet':
            self.net = MMDenseNet().to(args.device)
        elif args.model_name == 'mdensenet':
            self.net = MDenseNet().to(args.device)
        elif args.model_name == 'mdenselstm':
            self.net = MDenseLSTM().to(args.device)
        elif args.model_name == 'mmdenselstm':
            self.net = MMDenseLSTM().to(args.device)
        self.lr = args.lr
        # self.optim = optim.Adam(self.net.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
        # self.optim = optim.SGD(self.net.parameters(), lr=args.lr)
        self.optim = optim.RMSprop(self.net.parameters(), lr=args.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optim, step_size=10, gamma=0.8, last_epoch=-1, verbose=False)
        self.l2 = nn.MSELoss()
        self.bce = nn.BCELoss()
        self.cel = nn.CrossEntropyLoss()
        self.epoch = 0
        self.model_name = "test"
        self.save_dir = "./checkpoints/%s" % self.model_name
        self.logger = Logger('./log/%s/lr_%.5f' %
                             (self.model_name, self.lr))
        self.batch_size = args.batch_size

        if args.warm_start:
            self.load_model(args.checkpoint)
        self.log_model(self.net)


    def log_model(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

    def init_model(self, ):
        self.net.apply(gaussian_weights_init)

    def save(self, name):
        """Save the model"""
        save_dir = self.save_dir

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        state = {
            'net': self.net.state_dict(),
            'opt': self.optim.state_dict(),
            'epoch': self.epoch
        }

        torch.save(state, os.path.join(save_dir, name + '.pkl'))


    def reset_grad(self):
        """Zeros the gradient buffers."""
        self.net.zero_grad()


    def load_model(self, checkpoint):
        """Load the model checkpoints"""
        checkpoint = torch.load(checkpoint)
        self.net.load_state_dict(checkpoint['net'])
        self.optim.load_state_dict(checkpoint['opt'])
        self.epoch = checkpoint['epoch']


    def separate(self, mix, args, checkpoint):
        self.load_model(checkpoint)
        return separate(self.net, mix, args)

    def evaluate(self, mix, gt):
        """
        mix: test mixture (1, T)
        gt: test groundtruth (4, T)
        """
        insts = self.args.insts
        self.net.eval()
        predicts = separate(self.net, mix, self.args)
        ref, est = {}, {}
        for inst_idx, inst in enumerate(insts):
            ref[inst] = gt[inst_idx]
            est[inst] = predicts[inst_idx]
        ref = np.array([item for item in ref.values()])
        est = np.array([item for item in est.values()])
        sdr, sir, sar, _ = bss_eval_sources(ref, est)
        return sdr, sir, sar
    
    def spec_plot(self, mix, gt, iteration):
        """
        plot spectrogram and log in the tensorboard
        args:
            mix: [T]
            gt: [4, T]
        """
        spec_estimate, _, _ = self.compute_spec(mix)

        for inst_idx, inst in enumerate(self.args.insts):
            spec_gt = wave2spec(self.args, gt[inst_idx])[0]
            self.logger.log_img(spec_gt, spec_estimate[inst_idx], iteration, inst)

    def compute_spec(self, mix, ):

        self.net.eval()
        spec, phase, tmplen = wave2spec(self.args, mix)      
        freq, leng = spec.shape[0], spec.shape[1]
        tmp = np.zeros((freq, align(leng)), dtype=np.float32)
        spec = np.concatenate((spec, tmp), axis=1)

        spec = torch.cuda.FloatTensor(spec[None, None, :, :])
        inst_specs = self.net(spec)  # (1, 4, 512, T)

        spec_estimate = inst_specs[0, :, :, :leng].detach().numpy()   # (4, 512, T)

        return spec_estimate, phase, tmplen
