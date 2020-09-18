# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 12:16:39 2020

@author: ACTUS
"""

import numpy as np
import librosa
import torch
import librosa
import pickle
from synthesis import build_model
from synthesis import wavegen

metadata = pickle.load(open('metadata.pkl', "rb"))
print(metadata[0][1].shape, metadata[0][2].shape)

spect_vc = pickle.load(open('results.pkl', 'rb'))
device = torch.device("cuda")
model = build_model().to(device)
checkpoint = torch.load("checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])


file = 'p225xp225.wav' # doesn't work FUCK
file = 'p225_001.wav'

# test default one works
print(spect_vc[0][1].shape, spect_vc[-1][1].shape)
correct_waveform = wavegen(model, c=spect_vc[0][1])   
print(correct_waveform.shape)
librosa.output.write_wav(file+'baseline.wav', correct_waveform, sr=16000)



# test melcode, decode
hop_length = 256
y, sr = librosa.load('results/'+ file, sr = None)

## did they not use librosa to calculate melspectrogram

melSpec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=80, hop_length =hop_length, n_fft = 1024,
                                         fmin=125, fmax=7600)
print(melSpec.T.shape)
waveform = wavegen(model, c=melSpec.T)
print(waveform.shape)
librosa.output.write_wav(file+'_andback.wav', waveform, sr=16000)



melSpec_dB = librosa.power_to_db(melSpec, ref=np.max)



librosa.output.write_wav(file+'wtf.wav', y, sr=16000)


## sanity check
# convert pre-mel to file
waveform = wavegen(model, c=metadata[0][2])
print(waveform.shape)
librosa.output.write_wav(metadata[0][0]+'sanity.wav', waveform, sr=16000)
# works




