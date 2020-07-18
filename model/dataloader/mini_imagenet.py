import torch
import os.path as osp
from PIL import Image

from torchvision import transforms
from tqdm import tqdm
import numpy as np
from model.dataloader.transforms import *
from model.dataloader.base import BaseDataset
from model.utils import search_dir

THIS_PATH = osp.dirname(__file__)
ROOT_PATH = osp.abspath(osp.join(THIS_PATH, '..', '..'))
ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..', '..'))
dirs = ['/home/amax/data/mini-imagenet/images',
        '/home/lamda3/data/mini-imagenet/images',
        '/data/yangy/mini-imagenet/images',
        '/home/hanlu/data/mini-imagenet/images']
IMAGE_PATH1 = search_dir(dirs)
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
CACHE_PATH = osp.join(ROOT_PATH, '.cache/')


class MiniImageNet(BaseDataset):
    """ Usage:
    """

    @property
    def eval_setting(self):
        return [(5, 1), (5, 5), (5, 20), (5, 50)]

    @property
    def split_path(self):
        return SPLIT_PATH

    @property
    def cache_path(self):
        return CACHE_PATH

    #
    # # Transformation
    # if args.backbone_class == 'ConvNet':
    #     self.transform = transforms.Compose(
    #         transforms_list + [
    #             transforms.Normalize(np.array([0.485, 0.456, 0.406]),
    #                                  np.array([0.229, 0.224, 0.225]))
    #         ])
    # elif args.backbone_class == 'Res12':
    #     self.transform = transforms.Compose(
    #         transforms_list + [
    #             transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
    #                                  np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
    #         ])
    # elif args.backbone_class == 'Res18':
    #     self.transform = transforms.Compose(
    #         transforms_list + [
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #         ])
    # elif args.backbone_class == 'WRN':
    #     self.transform = transforms.Compose(
    #         transforms_list + [
    #             transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    #         ])
    # else:
    #     raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')

    def parse_csv(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label

    # def __len__(self):
    #     return len(self.data)
    #
    # def __getitem__(self, i):
    #     data, label = self.data[i], self.label[i]
    #
    #     if self.unsupervised and self.setname == 'train':
    #         image_list = []
    #         if not self.use_im_cache:
    #             data = Image.open(data).convert('RGB')
    #             # inp = self.flip_lr(data)
    #         if self.augment == 'AMDIM':
    #             data = self.flip_lr(data)
    #         for _ in range(self.repeat):
    #             image_list.append(self.transform(data))
    #             # inp = self.flip_lr(Image.open(data).convert('RGB'))
    #         return image_list, label
    #     else:
    #         if self.use_im_cache:
    #             image = self.transform(data)
    #         else:
    #             image = self.transform(Image.open(data).convert('RGB'))
    #
    #     return image, label
