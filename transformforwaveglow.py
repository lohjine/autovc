import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator
import pickle
import torch

speaker_emb_dim = 19
base = 32 # this is set to the last number i think


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

device = 'cuda:0'
G = Generator(32,speaker_emb_dim,512,16).eval().to(device) # 2nd number is  onehot

#g_checkpoint = torch.load('autovc.ckpt' ,map_location='cuda:0')

print('loading model')

g_checkpoint = torch.load('checkpoint/v5/chkpt_300000' ,map_location='cuda:0')
G.load_state_dict(g_checkpoint['model'])


# generate the metadata
# 

print('gen metadata')

metadata = []



rootDir = r'C:\Users\ACTUS\Desktop\pyscripts\autovc\data\autovc_train'
#rootDir = r'autovc_train'

musicDir = r'C:\Users\ACTUS\Desktop\pyscripts\autovc\data\autovc_train\\'
#musicDir = 'autovc_train/'

with open(os.path.join(rootDir, 'train.pkl'), 'rb') as handle:
    speakers = pickle.load(handle)
    
#for idx, i in enumerate(speakers):
#    print(idx, i[0])
    
# illya is 10
    
    
"""
dat = ['illya', speakers[10][1], np.load(musicDir+speakers[10][200])]
metadata.append(dat)

dat = ['renai', speakers[16][1], np.load(musicDir+speakers[16][13])]
metadata.append(dat)

dat = ['karen', speakers[11][1], np.load(musicDir+speakers[11][13])]
metadata.append(dat)

dat = ['platinum', speakers[14][1], np.load(musicDir+speakers[14][4])]
metadata.append(dat)



print('prediction')

spect_vc = []

for sbmt_i in metadata:
             
    x_org = sbmt_i[2]
    x_org, len_pad = pad_seq(x_org, base=base)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
    for sbmt_j in metadata:
                   
        emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
        spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        print('{}x{}'.format(sbmt_i[0], sbmt_j[0]))
        
        torch.save(torch.Tensor(uttr_trg.T), 'waveglowout/' + '{}x{}'.format(sbmt_i[0], sbmt_j[0]) + '.pt',
                   _use_new_zipfile_serialization=False)
        
with open('results.pkl', 'wb') as handle:
    pickle.dump(spect_vc, handle)    
    
"""

spect_vc = []
    
styles = []

dat = ['illya', speakers[10][1]]
styles.append(dat)

dat = ['karen', speakers[11][1]]
styles.append(dat)

targets = []    

dat = ['illya', speakers[10][1], np.load(musicDir+speakers[10][200])]
targets.append(dat)

for i in range(10,14):
    dat = [f'renai_{i}', speakers[16][1], np.load(musicDir+speakers[16][i])]
    targets.append(dat)
    
for i in range(3,7):
    dat = [f'platinum_{i}', speakers[14][1], np.load(musicDir+speakers[14][i])]
    targets.append(dat)
    
dat = ['renaifull', speakers[16][1], np.load(os.path.join(musicDir,'full','renai.wav.npy'))]
targets.append(dat)
    
dat = ['platfull', speakers[14][1], np.load(os.path.join(musicDir,'platinum','4.wav.npy'))]
targets.append(dat)


for target in targets:
    
    
    x_org = target[2]
    x_org, len_pad = pad_seq(x_org, base=base)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(target[1][np.newaxis, :]).to(device)
    
    
    for style in styles:
    
        emb_trg = torch.from_numpy(style[1][np.newaxis, :]).to(device)
        
        with torch.no_grad():
            _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
        if len_pad == 0:
            uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
        else:
            uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
        spect_vc.append( ('{}x{}'.format(target[0], style[0]), uttr_trg) )
        
        print('{}x{}'.format(target[0], style[0]))
        
        torch.save(torch.Tensor(uttr_trg.T), 'waveglowout/' + '{}x{}'.format(target[0], style[0]) + '.pt',
                   _use_new_zipfile_serialization=False)
        
    
    
print('complete')