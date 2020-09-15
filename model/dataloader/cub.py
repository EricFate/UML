import os.path as osp
import PIL
from PIL import Image
import torch

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from model.dataloader.base import BaseDataset
import os
from model.utils import search_dir

THIS_PATH = osp.dirname(__file__)
ROOT_PATH1 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..'))
dirs = ['/home/yehj/Few-Shot/data/cub/images',
        '/home/amax/data/cub/images',
        '/home/hanlu/data/cub/images']

IMAGE_PATH = search_dir(dirs)
SPLIT_PATH = osp.join(ROOT_PATH2, 'data/cub/split')
CACHE_PATH = osp.join(ROOT_PATH2, '.cache/')


# This is for the CUB dataset
# It is notable, we assume the cub images are cropped based on the given bounding boxes
# The concept labels are based on the attribute value, which are for further use (and not used in this work)


class CUB(BaseDataset):

    @property
    def eval_setting(self):
        return [(5, 1), (5, 5), (5, 20)]

    @property
    def split_path(self):
        return SPLIT_PATH

    @property
    def cache_path(self):
        return CACHE_PATH


    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]

        for l in lines:
            context = l.split(',')
            name = context[0]
            wnid = context[1]
            path = osp.join(IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1

            data.append(path)
            label.append(lb)

        return data, label

