from librosa import load, stft, istft
import torch
import numpy as np
from mir_eval.separation import bss_eval_sources
from hparams import create_hparams
import os
import utils
from models.model import Model
import json
from soundfile import write
from tqdm import tqdm
import argparse

class EVAL(object):

    def __init__(self, hparams, args):

        self.hparams = hparams
        self.model = Model(hparams)
        self.model_name = args.model
        self.model.load_model(args.checkpoint)

        self.model_list = [self.model]

    def eval(self,):
        os.makedirs(self.hparams.eval_path, exist_ok=True)
        test_metrics = []
        eval_file_list = os.listdir(os.path.join(self.hparams.wave_path, 'test'))
        for model in self.model_list:
            self.eval_model(model, eval_file_list, test_metrics)
        
        avg_SDRs = {inst: np.mean(np.nanmean(
            [song[inst]["sdr"] for song in test_metrics])) for inst in self.hparams.insts}
        avg_SIRs = {inst: np.mean(np.nanmean(
            [song[inst]["sir"] for song in test_metrics])) for inst in self.hparams.insts}
        avg_SARs = {inst: np.mean(np.nanmean(
            [song[inst]["sar"] for song in test_metrics])) for inst in self.hparams.insts}

        overall_SDR = np.mean([v for v in avg_SDRs.values()])
        print("%s SDR: %s" %(self.model_name, str(overall_SDR)))

    def eval_model(self, model, eval_file_list, test_metrics):
        print(f"Evaluating model: {self.model_name}...")
        for file in tqdm(eval_file_list):
            if os.path.exists(os.path.join(self.hparams.eval_path, self.model_name, file+'.json')):
                # print(f'{file} has been evaluated, skip...')
                continue
            self.evaluate(file, test_metrics, model)
            # print(f'{file} has been evaluated...')
        
    def evaluate(self, file, test_metrics, model):

        # os.makedirs(os.path.join(self.hparams.test_wav_estimate, file), exist_ok=True)
        # load test data
        par_dir = os.path.join(self.hparams.wave_path, 'test', file)
        mix, gt = utils.load_gt(self.hparams, par_dir)  # (T, ) (4, T)

        sdr, sir, sar = model.evaluate(mix, gt)

        song = {}
        for idx, inst in enumerate(self.hparams.insts):
            song[inst] = {
                "sdr": sdr[idx],
                "sir": sir[idx],
                "sar": sar[idx]
            }
        if not os.path.exists(os.path.join(self.hparams.eval_path, self.model_name)):
            os.mkdir(os.path.join(self.hparams.eval_path, self.model_name))
        with open(os.path.join(self.hparams.eval_path, self.model_name, file+'.json'), 'w') as f:
            json.dump(song, f)
        test_metrics.append(song)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # nohup python evaluate.py -c checkpoints/unet-baseline/best.pkl -m unet-baseline >out/eval.out &
    parser.add_argument("--checkpoint", "-c", default='checkpoints/mmdenselstm/best.pkl', type=str, help="model for evaluate")
    parser.add_argument("--model", "-m", default='mmdenselstm', type=str, help="model name")

    args = parser.parse_args()

    hparams = create_hparams()
    hparams.checkpoint = args.checkpoint
    
    eval = EVAL(hparams, args)
    eval.eval()
