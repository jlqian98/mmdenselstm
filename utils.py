import torch
import torch.nn as nn
import torchvision
import os
import pickle
import scipy.io
import numpy as np

from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from numpy.random import normal
from librosa.core import stft, load, istft, resample
from soundfile import write

def load_mix_gt(args, duration=60):
    file_dir = os.path.join(args.wave_path, "test", "Al James - Schoolboy Facination")
    mix, _ = load(os.path.join(file_dir, "mixture.wav"), sr=args.SR, duration=duration)
    gt = [load(os.path.join(file_dir, inst+'.wav'), sr=args.SR)[0] for inst in args.insts]
    return mix, gt

def gen_dcode(domain, batch, size):
    #domain=1 src; domain=0 tgt
    code=torch.rand(batch, 2, size[0], size[1])
    code[:,0,:,:]=domain
    code[:,1,:,:]=1-domain
    code=to_var(code)
    return code

def load_gt(C, file_dir, duration=None):
    """
    load groundtruth of the test mixture
    return: [drums, bass, other, vocal]
    """
    insts, gt = ['drums', 'bass', 'other', 'vocals', 'accompaniment', 'mixture'], []
    for inst in insts:
        if not duration is None:
            source, _ = load(os.path.join(file_dir, inst+'.wav'), sr=C.SR, duration=duration)
        else:
            source, _ = load(os.path.join(file_dir, inst+'.wav'), sr=C.SR)
        gt.append(source)
    return gt[-1], gt[:-1]


def spec_loss(out, sep, criterion):
    return criterion(out, sep)

def save_model(save_dir, model, model_name):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    # print("saving model in %s..." % save_path)
    torch.save(model, save_path)

def align(length):
    tmp = 2
    while length > tmp:
        tmp = tmp * 2
    return tmp - length

def wave2spec(C, y):
    tmp = []
    if (len(y) % C.H != 0):
        pad_size = C.H - len(y) % C.H
        tmp = np.zeros(pad_size)
        y = np.append(y, tmp)
    spec = stft(y, n_fft=C.fft_size, hop_length=C.H, win_length=C.fft_size)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j * np.angle(spec))
    return mag, phase, len(tmp)

def spec2wave(C, mag, phase, tmplen):
    y = istft(mag * phase, hop_length=C.H, win_length=C.fft_size)
    y = y[:len(y) - tmplen]
    return y / np.max(np.abs(y))


def LoadAudio(C, fname):
    # load mono music
    y, sr = load(fname, sr=C.SR)
    tmp = []
    if (len(y) % C.H != 0):
        pad_size = C.H - len(y) % C.H
        tmp = np.zeros(pad_size)
        y = np.append(y, tmp)
    spec = stft(y, n_fft=C.fft_size, hop_length=C.H, win_length=C.fft_size)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j * np.angle(spec))
    return mag, phase, len(tmp)

def SaveAudio(C, fname, mag, phase, tmplen):
    y = istft(mag * phase, hop_length=C.H, win_length=C.fft_size)
    y = y[:len(y) - tmplen]
    write(fname, (y / np.max(np.abs(y))), C.SR)



def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=False)
    
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.save(val_loss, score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.save(val_loss, score, model)
            self.counter = 0

    def save(self, val_loss, score, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.10f} --> {val_loss:.10f}).  Saving model ...')
        model.save("best")
        self.val_loss_min = val_loss
        self.best_score = score


def scale_data(feature, label, augmentation):
    label_scaled = []
    if augmentation:
        scale_factor = random.choice([x / 100.0 for x in range(80, 124, 4)])
        if scale_factor != 1:
            feature_scaled = apply_affine_transform(feature, zy=scale_factor, fill_mode='nearest').astype(np.float32)
            for row in range(len(label)):
                label_scaled.append(
                    [round(label[row][0] * scale_factor), round(label[row][1] * scale_factor)])
        else:
            feature_scaled = feature
            label_scaled = label
    else:
        feature_scaled = feature
        label_scaled = label
    return feature_scaled, label_scaled


def pitch_shift_spectrogram(feature):
    """ Shift a spectrogram along the frequency axis in the spectral-domain at
    random
    """
    nb_cols = feature.shape[0]
    max_shifts = nb_cols // 20 # around 5% shift
    nb_shifts = np.random.randint(-max_shifts, max_shifts)

    return np.roll(feature, nb_shifts, axis=0)