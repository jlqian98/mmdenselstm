import argparse
import librosa
import utils
import os
import numpy as np
from models.model import Model
from utils import align
from hparams import create_hparams
from soundfile import write
import torch

MAX_INT16 = np.iinfo(np.int16).max
hparams = create_hparams()

class SEP(object):

    def __init__(self, haprams, args):
        self.hparams = hparams
        self.test_dir = args.mix_dir
        self.model_name = args.model
        self.model_path = args.checkpoint
        self.audio_list = os.listdir(self.test_dir)
        self.model = Model(hparams)
        self.model.load_model(self.model_path)

    
    def sep(self, ):
        for fname in self.audio_list:
            self.separate(fname)

    def separate(self, fname):
        mix, _ = librosa.load(os.path.join(self.test_dir, fname), sr=self.hparams.SR)
        insts_wav = self.model.separate(mix, self.hparams, self.model_path)

        insts = ['drums', 'bass', 'other', 'vocals', 'accompaniment']
        for idx, inst in enumerate(insts):
            path_dir = os.path.join(self.hparams.separate_path, fname.split('.')[0], self.model_name)
            os.makedirs(path_dir, exist_ok=True)
            write(os.path.join(path_dir, inst+'.wav'), insts_wav[idx], self.hparams.SR)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # python separate.py --checkpoint checkpoints/unet-test/best.pkl --mix_dir mix_test
    parser.add_argument("--checkpoint", default='', type=str, help="model for separate")
    parser.add_argument("--mix_dir", default='', type=str, help="dir of mixture")
    parser.add_argument("--model", default='unet', type=str, help="model name")

    args = parser.parse_args()

    hparams = create_hparams()
    # TODO 进一步封装
    hparams.checkpoint = args.checkpoint
    sep = SEP(hparams, args)
    sep.sep()
