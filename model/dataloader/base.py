import os.path as osp
import PIL
from PIL import Image
import torch

import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from model.dataloader.transforms import *
import moco.loader

_imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


def identity(x):
    return x


class RandomTranslateWithReflect:
    '''
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class BaseDataset(Dataset):
    def __init__(self, setname, unsupervised, args, augment='none'):
        im_size = args.orig_imsize
        csv_path = osp.join(self.split_path, setname + '.csv')
        cache_path = osp.join(self.cache_path, "{}.{}.{}.pt".format(self.__class__.__name__, setname, im_size))

        self.unsupervised = unsupervised
        self.setname = setname
        if hasattr(args, 'shot') and hasattr(args, 'query'):
            self.repeat = args.shot + args.query
        else:
            self.repeat = 1
        self.augment = augment

        self.use_im_cache = (im_size != -1)  # not using cache
        if self.use_im_cache:
            if not osp.exists(cache_path):
                print('* Cache miss... Preprocessing {}...'.format(setname))
                resize_ = identity if im_size < 0 else transforms.Resize(im_size)
                data, label = self.parse_csv(csv_path)
                self.data = [resize_(Image.open(path).convert('RGB')) for path in data]
                self.label = label
                print('* Dump cache from {}'.format(cache_path))
                torch.save({'data': self.data, 'label': self.label}, cache_path)
            else:
                print('* Load cache from {}'.format(cache_path))
                cache = torch.load(cache_path)
                self.data = cache['data']
                self.label = cache['label']
        else:
            self.data, self.label = self.parse_csv(csv_path)

        self.num_class = len(set(self.label))

        image_size = 84
        if augment != 'none' and setname == 'train':
            if self.unsupervised:
                if augment == 'AMDIM':
                    self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
                    col_jitter = transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
                    img_jitter = transforms.RandomApply([RandomTranslateWithReflect(4)], p=0.8)
                    rnd_gray = transforms.RandomGrayscale(p=0.25)

                    # self.transform = transforms.Compose([
                    #     transforms.RandomResizedCrop(image_size),
                    #     img_jitter,
                    #     col_jitter,
                    #     rnd_gray,
                    #     transforms.ToTensor(),
                    #     transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                    #                          np.array([0.229, 0.224, 0.225])),
                    #
                    # ])
                    transforms_list = [
                        transforms.RandomResizedCrop(image_size),
                        # flip_lr,
                        img_jitter,
                        col_jitter,
                        rnd_gray,
                        transforms.ToTensor(),
                    ]
                elif augment == 'SimCLR':
                    s = 1
                    color_jitter = transforms.ColorJitter(
                        0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
                    )
                    transforms_list = [
                        transforms.RandomResizedCrop(size=96),
                        transforms.RandomHorizontalFlip(),  # with 0.5 probability
                        transforms.RandomApply([color_jitter], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.ToTensor(),
                    ]
                elif augment == 'AutoAug':
                    from .autoaug import RandAugment
                    transforms_list = [
                        RandAugment(2, 12),
                        ERandomCrop(image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(0.4, 0.4, 0.4),
                        transforms.ToTensor(),
                        Lighting(0.1, _imagenet_pca['eigval'], _imagenet_pca['eigvec']),
                    ]

                else:
                    raise ValueError('Non-supported Augmentation Types. Please Revise Data Pre-Processing Scripts.')
            else:
                if augment == 'moco':
                    transforms_list = [
                        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.)),
                        transforms.RandomApply([
                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                        ], p=0.8),
                        transforms.RandomGrayscale(p=0.2),
                        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                else:
                    transforms_list = [
                        transforms.RandomResizedCrop(image_size),
                        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
        else:
            transforms_list = [
                transforms.Resize(92),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
            ]
        # Transformation
        if args.backbone_class == 'ConvNet':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                         np.array([0.229, 0.224, 0.225]))
                ])
        elif args.backbone_class == 'Res12':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                         np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
                ])
        elif args.backbone_class == 'Res18':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        elif args.backbone_class == 'WRN':
            self.transform = transforms.Compose(
                transforms_list + [
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                ])
        else:
            raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
        if augment == 'moco':
            self.transform = moco.loader.TwoCropsTransform(self.transform)
        self.image_shape = self.__getitem__(0)[0][0].shape

    @property
    def split_path(self):
        raise NotImplementedError

    @property
    def cache_path(self):
        raise NotImplementedError

    @property
    def eval_setting(self):
        raise NotImplementedError

    def parse_csv(self, csv_path):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    # @profile
    def __getitem__(self, i):
        data, label = self.data[i], self.label[i]

        if self.unsupervised and self.setname == 'train':
            image_list = []
            if not self.use_im_cache:
                data = Image.open(data).convert('RGB')
                # inp = self.flip_lr(data)
            if self.augment == 'AMDIM':
                data = self.flip_lr(data)
            for _ in range(self.repeat):
                image_list.append(self.transform(data))
                # inp = self.flip_lr(Image.open(data).convert('RGB'))
            return image_list, label
        else:
            if self.use_im_cache:
                image = self.transform(data)
            else:
                image = self.transform(Image.open(data).convert('RGB'))

        return image, label
