import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
import pickle
import torch
import torch.nn.functional as F

speaker_emb_dim = 19
lambda_cd = 1

def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'
G = Generator(32,speaker_emb_dim,512,32).eval().to(device) # 2nd number is  onehot

#g_checkpoint = torch.load('autovc.ckpt' ,map_location='cuda:0')

print('loading model')

g_checkpoint = torch.load('checkpoint/chkpt_340000' ,map_location='cuda:0')
G.load_state_dict(g_checkpoint['model'])


# generate the metadata
#

print('gen metadata')

metadata = []



#rootDir = r'C:\Users\ACTUS\Desktop\pyscripts\autovc\data\autovc_train'
rootDir = r'autovc_train'

#musicDir = r'C:\Users\ACTUS\Desktop\pyscripts\autovc\data\autovc_train\\'
musicDir = 'autovc_train/'

with open(os.path.join(rootDir, 'train.pkl'), 'rb') as handle:
    speakers = pickle.load(handle)


errors = []

for speaker in speakers:

    emb_org = torch.from_numpy(speaker[1][np.newaxis, :]).to(device)

    for sample in speaker[2:]:

        x_org = np.load(musicDir+sample)
        x_org, len_pad = pad_seq(x_org)

        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

        # no, look at solver_encoder.py to do this part
        # to calc error


        with torch.no_grad():
            x_identic, x_identic_psnt, code_real = G(uttr_org, emb_org, emb_org)

        # Identity mapping loss
        g_loss_id = F.mse_loss(uttr_org, x_identic)
        g_loss_id_psnt = F.mse_loss(uttr_org, x_identic_psnt)

        # Code semantic loss.
        code_reconst = G(x_identic_psnt, emb_org, None)
        g_loss_cd = F.l1_loss(code_real, code_reconst)

#        g_loss = g_loss_id + g_loss_id_psnt + lambda_cd * g_loss_cd

        errors.append(( speaker[0] , sample , g_loss_cd.item()))



errors = sorted(errors, key=lambda x: x[2].item(), reverse=True)

for i in errors[:20]:
    print(i)

print('complete')