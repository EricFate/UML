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
group = parser.add_mutually_exclusive_group()
group.add_argument("-v", "--verbose", action="store_true")
group.add_argument("-q", "--quiet", action="store_true")
parser.add_argument("--name", type=str, default='train')
parser.add_argument("--dataset", type=str, default='miniimagenet')
parser.add_argument('-s', "--shot", type=int, default=0)
parser.add_argument('-w', "--way", type=int, default=0)
args = parser.parse_args()

dir_path = os.path.join('data', args.dataset, 'split')
file = os.path.join(dir_path, '%s.csv' % args.name)

df = pd.read_csv(file)
labels = df.loc[:, 'label']
label_unique = pd.unique(labels)
data = []

if args.way > 0:
    label_unique = np.random.choice(label_unique, args.way)

for l in label_unique:
    datum = df[labels == l]
    if args.shot > 0:
        idxes = np.random.choice(len(datum), args.shot, replace=False)
        idxes = sorted(idxes)
    else:
        idxes = np.arange(len(datum))
    data.append(datum.iloc[idxes, :])
print(len(data))
data = pd.concat(data, axis=0)
print(len(data))
file_name = '%s_%d_%d.csv' % (args.name, args.way, args.shot)
save_path = os.path.join(dir_path, file_name)
data.to_csv(save_path, index=False)
