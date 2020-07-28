# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 15:16:36 2020

@author: hanlu
"""

import numpy as np
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='train')
parser.add_argument("--dataset", type=str, default='miniimagenet')
parser.add_argument("--number", type=int, default=50)
args = parser.parse_args()

dir_path = os.path.join('data', args.dataset, 'split')
file = os.path.join(dir_path, '%s.csv' % args.name)

df = pd.read_csv(file)
labels = df.loc[:, 'label']
label_unique = pd.unique(labels)
data = []

for l in label_unique:
    datum = df[labels == l]
    idxes = np.random.choice(len(datum), args.number,replace=False)
    idxes = sorted(idxes)
    data.append(datum.iloc[idxes, :])
print(len(data))
data = pd.concat(data, axis=0)
print(len(data))
file_name = '%s_%d.csv' % (args.name, args.number)
save_path = os.path.join(dir_path, file_name)
data.to_csv(save_path, index=False)
