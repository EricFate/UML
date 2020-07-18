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
# IMAGE_PATH = osp.join(ROOT_PATH1, 'data/cub')
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

    # def __init__(self, setname, , args, augment=False):
    #     im_size = args.orig_imsize
    #     txt_path = osp.join(SPLIT_PATH, setname + '.csv')
    #     # lines = [x.strip() for x in open(txt_path, 'r').readlines()][1:]
    #     cache_path = osp.join(CACHE_PATH, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size))
    #
    #     self.use_im_cache = (im_size != -1)  # not using cache
    #     if self.use_im_cache:
    #         if not osp.exists(cache_path):
    #             print('* Cache miss... Preprocessing {}...'.format(setname))
    #             resize_ = identity if im_size < 0 else transforms.Resize(im_size)
    #             data, label = self.parse_csv(txt_path)
    #             self.data = [resize_(Image.open(path).convert('RGB')) for path in data]
    #             self.label = label
    #             print('* Dump cache from {}'.format(cache_path))
    #             torch.save({'data': self.data, 'label': self.label}, cache_path)
    #         else:
    #             print('* Load cache from {}'.format(cache_path))
    #             cache = torch.load(cache_path)
    #             self.data = cache['data']
    #             self.label = cache['label']
    #     else:
    #         self.data, self.label = self.parse_csv(txt_path)
    #
    #     self.num_class = np.unique(np.array(self.label)).shape[0]
    #     image_size = 84
    #
    #     if augment != 'none' and setname == 'train':
    #         if self.unsupervised:
    #             if augment == 'AMDIM':
    #                 self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
    #                 col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
    #                 img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
    #                 rnd_gray = transforms.RandomGrayscale(p=0.25)
    #
    #                 # self.transform = transforms.Compose([
    #                 #     transforms.RandomResizedCrop(image_size),
    #                 #     img_jitter,
    #                 #     col_jitter,
    #                 #     rnd_gray,
    #                 #     transforms.ToTensor(),
    #                 #     transforms.Normalize(np.array([0.485, 0.456, 0.406]),
    #                 #                          np.array([0.229, 0.224, 0.225])),
    #                 #
    #                 # ])
    #                 transforms_list = [
    #                     transforms.RandomResizedCrop(image_size),
    #                     # flip_lr,
    #                     img_jitter,
    #                     col_jitter,
    #                     rnd_gray,
    #                     transforms.ToTensor(),
    #                 ]
    #             elif augment == 'SimCLR':
    #                 s = 1
    #                 color_jitter = transforms.ColorJitter(
    #                     0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
    #                 )
    #                 transforms_list = [
    #                     transforms.RandomResizedCrop(size=96),
    #                     transforms.RandomHorizontalFlip(),  # with 0.5 probability
    #                     transforms.RandomApply([color_jitter], p=0.8),
    #                     transforms.RandomGrayscale(p=0.2),
    #                     transforms.ToTensor(),
    #                 ]
    #             elif augment == 'AutoAug':
    #                 from .autoaug import RandAugment
    #                 transforms_list = [
    #                     RandAugment(2, 12),
    #                     ERandomCrop(image_size),
    #                     transforms.RandomHorizontalFlip(),
    #                     transforms.ColorJitter(0.4, 0.4, 0.4),
    #                     transforms.ToTensor(),
    #                     Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
    #                 ]
    #
    #             else:
    #                 raise ValueError('Non-supported Augmentation Types. Please Revise Data Pre-Processing Scripts.')
    #         else:
    #             transforms_list = [
    #                 transforms.RandomResizedCrop(image_size),
    #                 transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    #                 transforms.RandomHorizontalFlip(),
    #                 transforms.ToTensor(),
    #             ]
    #     else:
    #         transforms_list = [
    #             transforms.Resize(92),
    #             transforms.CenterCrop(image_size),
    #             transforms.ToTensor(),
    #         ]

    def parse_csv(self, txt_path):
        data = []
        label = []
        lb = -1
        self.wnids = []
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

    # def __len__(self):
    #     return len(self.data)
    #
    # def __getitem__(self, i):
    #     data, label = self.data[i], self.label[i]
    #     if self.use_im_cache:
    #         image = self.transform(data)
    #     else:
    #         image = self.transform(Image.open(data).convert('RGB'))
    #     return image, label
