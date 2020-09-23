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

G = Generator(64,speaker_emb_dim,512,16).eval().to(device) # 2nd number is  onehot

#g_checkpoint = torch.load('autovc.ckpt' ,map_location='cuda:0')

print('loading model')

g_checkpoint = torch.load('checkpoint/chkpt_340000' ,map_location='cuda:0')
G.load_state_dict(g_checkpoint['model'])




#rootDir = r'C:\Users\ACTUS\Desktop\pyscripts\autovc\data\autovc_train'
rootDir = r'autovc_train'

#musicDir = r'C:\Users\ACTUS\Desktop\pyscripts\autovc\data\autovc_train\\'
musicDir = 'autovc_train/'

with open(os.path.join(rootDir, 'train.pkl'), 'rb') as handle:
    speakers = pickle.load(handle)


content = []

for idx, speaker in enumerate(speakers):

    emb_org = torch.from_numpy(speaker[1][np.newaxis, :]).to(device)

    for sample in speaker[2:]:

        x_org = np.load(musicDir+sample)
        x_org, len_pad = pad_seq(x_org)

        uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)

        # no, look at solver_encoder.py to do this part
        # to calc error

        with torch.no_grad():
            _, _, code_real = G(uttr_org, emb_org, emb_org)

        ## dump code_real into savefile for classification error training/testing
        content.append(np.concatenate(code_real.toarray(), idx)) # and append class

        raise # see what code_real looks like, flatten it?

        # if array dim not huge, can just use RF

## === split here to see what code_real / content looks like

from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import classification_report

#final_array[:,:-1] = preprocessing.StandardScaler().fit_transform(final_array[:,:-1])

skf = StratifiedKFold(n_splits=5)

train_scores = []
test_scores = []

train_acc = []
test_acc = []

for train_index, test_index in skf.split(content[:,:-1], content[:,-1]):

    x_train, x_test = content[train_index,:-1], content[test_index,:-1]
    y_train, y_test = content[train_index,-1], content[test_index,-1]

    clf = RandomForestClassifier(n_estimators=300, verbose = 0, n_jobs=3)
    clf.fit(x_train, y_train)

    train_scores.append(f1_score(y_train, clf.predict(x_train),multi_class='ovo'))
    test_scores.append(f1_score(y_test, clf.predict(x_test),multi_class='ovo'))

    train_acc.append(accuracy_score(y_train, clf.predict(x_train)))
    test_acc.append(accuracy_score(y_test, clf.predict(x_test)))


print('train f1 score:', np.mean(train_scores), ' ', np.std(train_scores))
print('test f1 score:', np.mean(test_scores), ' ', np.std(test_scores))

print('train_acc score:', np.mean(train_acc), ' ', np.std(train_acc))
print('test_acc score:', np.mean(test_acc), ' ', np.std(test_acc))

# target classification accuracy is ~12% ?

## unbalanced classes
# need to see confusion matrix
#print(classification_report(y_test, clf.predict(x_test)))

