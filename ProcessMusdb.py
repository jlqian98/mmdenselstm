#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:49:54 2017

@author: wuyiming
"""

import numpy as np
from librosa.core import load, resample, stft
import os.path
from tqdm import tqdm
from hparams import create_hparams

def save_spectrogram_inst(C, y_mix, y_insts, fname, original_sr=44100):

    if not os.path.exists(C.fft_path):
        os.mkdir(C.fft_path)
    path_train = os.path.join(C.fft_path, 'train')
    path_valid = os.path.join(C.fft_path, 'valid')
    os.makedirs(path_train, exist_ok=True)
    os.makedirs(path_valid, exist_ok=True)


    y_mix = resample(y_mix, original_sr, C.SR)
    y_insts = [resample(y_inst, original_sr, C.SR) for y_inst in y_insts]
    y_insts = np.array(y_insts)

    s_mix = np.abs(
        stft(y_mix, n_fft=C.fft_size, hop_length=C.H)).astype(np.float32)
    norm = s_mix.max()
    s_mix = s_mix / norm
    s_insts = []
    for y_inst in y_insts:
        s_inst = np.abs(stft(y_inst, n_fft=C.fft_size, hop_length=C.H).astype(np.float32))
        s_inst /= norm
        s_insts.append(s_inst)
    s_insts = np.stack(s_insts, axis=0)

    # Generate sequence (1,512,128) and save
    total_cnt = s_mix.shape[1] // C.patch_length
    cnt = 1
    i = 0
    while i + C.patch_length < s_mix.shape[1]:
        mix_spec = s_mix[None, :, i:i + C.patch_length]  # (1, 512, 128)
        inst_spec = s_insts[:, :, i:i + C.patch_length]  # (4, 512, 128)
        if cnt <= int(total_cnt * C.training_rate):
            np.savez(os.path.join(path_train, fname + str(cnt) + "_4s" + ".npz"), data=mix_spec, label=inst_spec)
        else:
            np.savez(os.path.join(path_valid, fname + str(cnt) + "_4s" + ".npz"), data=mix_spec, label=inst_spec)
        i += C.patch_length
        cnt += 1

    if s_mix.shape[1] >= 128:
        mix_spec = s_mix[None, :, s_mix.shape[1] - C.patch_length:s_mix.shape[1]]
        inst_spec = s_insts[:, :, s_mix.shape[1] - C.patch_length:s_mix.shape[1]]

        np.savez(os.path.join(path_valid, fname + str(cnt) + "_4s" + ".npz"), data=mix_spec, label=inst_spec)


def save_to_spec_inst(C, convert_path):
    """
    Wave data -> spec data (four sources)
    :param convert_path: wave dataset path
    """
    print('Start to convert the wave data to fft data...')
    music_source = [os.path.join(convert_path, f) for f in os.listdir(convert_path)]
    for music_dir in tqdm(music_source):
        if music_dir.endswith(".DS_Store"):
            continue
        # print('Processing: ' + music_dir)
        y_mix, _ = load(os.path.join(music_dir, 'mixture.wav'), sr=None)
        y_insts = []
        insts = C.insts
        for inst in insts:
            y_inst, _ = load(os.path.join(music_dir, inst + '.wav'), sr=None)
            y_insts.append(y_inst)
        y_insts = np.stack(y_insts, axis=0)  # (T, ), (C, T)
        file_mix_name = music_dir.split("/")[-1]
        # file_mix_name, _ = os.path.splitext(music_dir)
        save_spectrogram_inst(C, y_mix, y_insts, file_mix_name)
    print('Convert finished!')

hparams = create_hparams()
save_to_spec_inst(hparams, os.path.join(hparams.wave_path, 'train'))



# def save_to_spec(convert_path):
#     """
#     Wave data -> spec data (vocals and accompaniment)
#     :param convert_path: wave dataset path
#     """
#     print('Start to convert the wave data to fft data...')
#     music_source = [os.path.join(convert_path, f) for f in os.listdir(convert_path)]
#     for music_dir in music_source:
#         if music_dir.endswith('.DS_Store'):
#             continue
#         # print("Processing: " + music_dir)
#         y_ins, _ = load(os.path.join(music_dir, "acc.wav"), sr=None)
#         y_vocal, _ = load(os.path.join(music_dir, "vocals.wav"), sr=None)
#         y_mix = y_ins + y_vocal
#         assert (y_mix.shape == y_vocal.shape)
#         assert (y_mix.shape == y_ins.shape)
#         file_mix_name = music_dir.split("/")[-1]
#         SaveSpectrogram(y_mix, y_vocal, y_ins, file_mix_name)
#     print('Convert finished!')
