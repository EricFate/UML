import argparse
import os.path as osp
import shutil
import time
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
# from model.nn.loss import LabelSmoothing, NLLMultiLabelSmooth
# from model.utils import pprint, set_gpu, mkdir, Averager, Timer, count_acc, euclidean_metric, MixUpWrapper
from model.utils import pprint, set_gpu, Averager, Timer, count_acc, euclidean_metric

from torch.utils.data import Dataset
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
from torchvision import transforms
from find_optim_test_size import get_path, get_loader


def search_dir(dirs):
    data_dir = None

    for d in dirs:
        if os.path.exists(d) and os.path.isdir(d):
            data_dir = d
            break
    if data_dir is None:
        raise FileNotFoundError('Data directory not found')
    print('data directory : %s' % data_dir)
    return data_dir


THIS_PATH = osp.dirname(__file__)
ROOT_PATH = THIS_PATH
SPLIT_PATH = osp.join(ROOT_PATH, 'data/miniimagenet/split')
# ROOT_PATH2 = osp.abspath(osp.join(THIS_PATH, '..', '..', '..', '..'))
dirs = ['/home/amax/data/mini-imagenet/images',
        '/home/lamda3/data/mini-imagenet/images',
        '/data/yangy/mini-imagenet/images',
        '/home/hanlu/data/mini-imagenet/images']
IMAGE_PATH1 = search_dir(dirs)


class MiniImageNet(Dataset):

    def __init__(self, setname, transform=None, target_transform=None) -> None:
        self.wnids = []
        csv_path = osp.join(self.split_path, setname + '.csv')
        self.data, self.labels = self.parse_csv(csv_path)
        self.num_class = np.unique(self.labels).__len__()
        self.transform = transform
        self.target_transform = target_transform

    @property
    def split_path(self):
        return SPLIT_PATH

    def parse_csv(self, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        for l in tqdm(lines, ncols=64):
            name, wnid = l.split(',')
            path = osp.join(IMAGE_PATH1, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        return data, label

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        img = Image.open(img).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


def get_transform(args):
    transforms_list = [
        transforms.Resize(int((92 / 84) * args.test_size)),
        transforms.CenterCrop(args.test_size),
        transforms.ToTensor(),
    ]

    # Transformation
    if args.backbone_class == 'ConvNet':
        transform = transforms.Compose(
            transforms_list + [
                transforms.Normalize(np.array([0.485, 0.456, 0.406]),
                                     np.array([0.229, 0.224, 0.225]))
            ])
    elif args.backbone_class == 'Res12':
        transform = transforms.Compose(
            transforms_list + [
                transforms.Normalize(np.array([x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]),
                                     np.array([x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]))
            ])
    elif args.backbone_class == 'Res18':
        transform = transforms.Compose(
            transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    elif args.backbone_class == 'WRN':
        transform = transforms.Compose(
            transforms_list + [
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
    else:
        raise ValueError('Non-supported Network Types. Please Revise Data Pre-Processing Scripts.')
    return transform


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImagenet'])
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'ConvAttNet', 'Res12',
                                                                                'Res18', 'ResNest12', 'SAN'])
    parser.add_argument('--query', type=int, default=15)

    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label-smoothing (default eta: 0.0)')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup (default eta: 0.0)')
    parser.add_argument('--rand-aug', action='store_true', default=False, help='random augment')
    parser.add_argument('--augment', type=str, default='none')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0008)
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler (default: cos)')
    parser.add_argument('--step', type=str, default='[2,3,4,5]', help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--no-bn-wd', action='store_true', default=False, help='no bias decay')
    parser.add_argument('--warmup-epochs', type=int, default=0, help='number of warmup epochs (default: 0)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='SGD weight decay (default: 1e-4)')

    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--samples_per_class', type=int, default=50)
    parser.add_argument('--num_test_episodes', type=int, default=2000)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--init_weights', type=str, default='./checkpoints/best/Res12-pre.pth')
    parser.add_argument('--test_size', type=int, default=84)
    parser.add_argument('--eval_interval', type=int, default=1)
    args = parser.parse_args()
    args.orig_imsize = -1
    pprint(vars(args))
    set_gpu(args.gpu)

    return args


def get_optimizer(args, model, train_loader):
    if args.no_bn_wd:
        bn_params = [v for n, v in model.named_parameters() if ('bn' in n or 'bias' in n)]
        rest_params = [v for n, v in model.named_parameters() if not ('bn' in n or 'bias' in n)]
        optimizer = torch.optim.SGD([{'params': bn_params, 'weight_decay': 0},
                                     {'params': rest_params, 'weight_decay': args.weight_decay}],
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.lr,
                                    momentum=0.9,
                                    weight_decay=args.weight_decay)

    if args.lr_scheduler == 'cosine':
        lr_schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_epoch * len(train_loader))
    elif args.lr_scheduler == 'step':
        lr_schedule = torch.optim.lr_scheduler.StepLR(optimizer, int(args.step) * len(train_loader), args.gamma)
    elif args.lr_scheduler == 'multistep':
        lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer, [e * len(train_loader) for e in eval(args.step)],
                                                           args.gamma)
    else:
        raise ValueError('No Such Schedule')

    if args.warmup_epochs > 0:
        lr_schedule_warmup = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0 * epoch / (
                args.warmup_epochs * len(train_loader)))
    else:
        lr_schedule_warmup = None

    return optimizer, lr_schedule, lr_schedule_warmup


def save_model(name, model, args):
    torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))


def save_checkpoint(is_best, epoch, filename='checkpoint.pth.tar'):
    state = {'epoch': epoch + 1,
             'args': args,
             'state_dict': model.state_dict(),
             'trlog': trlog,
             'val_acc_dist': trlog['max_acc_dist'],
             'val_acc_sim': trlog['max_acc_sim'],
             'optimizer': optimizer.state_dict(),
             'global_count': global_count}

    torch.save(state, osp.join(args.save_path, filename))
    if is_best:
        shutil.copyfile(osp.join(args.save_path, filename), osp.join(args.save_path, 'model_best.pth.tar'))


def train(args, model, train_loader, criterion, optimizer, lr_schedule, trlog):
    model.train()
    tl = Averager()
    ta = Averager()
    global global_count, writer
    for i, (data, label) in enumerate(train_loader, 1):
        global_count = global_count + 1
        if torch.cuda.is_available():
            data, label = data.cuda(), label.long().cuda()
        elif not args.mixup:
            label = label.long()

        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        lr_schedule.step()

        writer.add_scalar('data/loss', float(loss), global_count)
        tl.add(loss.item())
        if not args.mixup:
            acc = count_acc(logits, label)
            ta.add(acc)
            writer.add_scalar('data/acc', float(acc), global_count)

        if (i - 1) % 100 == 0 or i == len(train_loader):
            if not args.mixup:
                print('epoch {}, train {}/{}, lr={:.5f}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader),
                                                                                        optimizer.param_groups[0]['lr'],
                                                                                        loss.item(), acc))
            else:
                print('epoch {}, train {}/{}, lr={:.5f}, loss={:.4f}'.format(epoch, i, len(train_loader),
                                                                             optimizer.param_groups[0]['lr'],
                                                                             loss.item()))

    if trlog is not None:
        tl = tl.item()
        trlog['train_loss'].append(tl)
        if not args.mixup:
            ta = ta.item()
            trlog['train_acc'].append(ta)
        else:
            trlog['train_acc'].append(0)
        return model, trlog
    else:
        return model


def validate(args, model, val_loader, epoch, trlog=None):
    model.eval()
    global writer
    vl_dist, va_dist, vl_sim, va_sim = Averager(), Averager(), Averager(), Averager()
    if trlog is not None:
        print('[Dist] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_dist_epoch'],
                                                                         trlog['max_acc_dist']))
        print(
            '[Sim] best epoch {}, current best val acc={:.4f}'.format(trlog['max_acc_sim_epoch'], trlog['max_acc_sim']))
    # test performance with Few-Shot
    label = torch.arange(args.num_val_class).repeat(args.query).long()
    if torch.cuda.is_available():
        label = label.cuda()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader, 1), total=len(val_loader)):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data, _ = batch
            data_shot, data_query = data[:args.num_val_class], data[args.num_val_class:]  # 16-way test
            logits_dist, logits_sim = model.forward_proto(data_shot, data_query, args.num_val_class)
            loss_dist = F.cross_entropy(logits_dist, label)
            acc_dist = count_acc(logits_dist, label)
            loss_sim = F.cross_entropy(logits_sim, label)
            acc_sim = count_acc(logits_sim, label)
            vl_dist.add(loss_dist.item())
            va_dist.add(acc_dist)
            vl_sim.add(loss_sim.item())
            va_sim.add(acc_sim)

    vl_dist = vl_dist.item()
    va_dist = va_dist.item()
    vl_sim = vl_sim.item()
    va_sim = va_sim.item()

    print(
        'epoch {}, val, loss_dist={:.4f} acc_dist={:.4f} loss_sim={:.4f} acc_sim={:.4f}'.format(epoch, vl_dist, va_dist,
                                                                                                vl_sim, va_sim))

    if trlog is not None:
        writer.add_scalar('data/val_loss_dist', float(vl_dist), epoch)
        writer.add_scalar('data/val_acc_dist', float(va_dist), epoch)
        writer.add_scalar('data/val_loss_sim', float(vl_sim), epoch)
        writer.add_scalar('data/val_acc_sim', float(va_sim), epoch)

        if va_dist > trlog['max_acc_dist']:
            trlog['max_acc_dist'] = va_dist
            trlog['max_acc_dist_epoch'] = epoch
            save_model('max_acc_dist', model, args)
            save_checkpoint(True, epoch)

        if va_sim > trlog['max_acc_sim']:
            trlog['max_acc_sim'] = va_sim
            trlog['max_acc_sim_epoch'] = epoch
            save_model('max_acc_sim', model, args)
            save_checkpoint(True, epoch)

        trlog['val_loss_dist'].append(vl_dist)
        trlog['val_acc_dist'].append(va_dist)
        trlog['val_loss_sim'].append(vl_sim)
        trlog['val_acc_sim'].append(va_sim)
        return trlog


if __name__ == '__main__':
    args = get_args()
    get_path(args)
    train_loader, val_loader = get_loader(args)
    model = Classifier(args)
    # if args.mixup > 0:
    #     train_loader = MixUpWrapper(args.mixup, args.num_class, train_loader)
    #     criterion = NLLMultiLabelSmooth(args.label_smoothing)
    # elif args.label_smoothing > 0.0:
    #     criterion = LabelSmoothing(args.label_smoothing)
    # else:
    # load model state
    model_dict = model.state_dict()
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights)['params']
        # remove weights for FC
        # pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer, lr_schedule, lr_schedule_warmup = get_optimizer(args, model, train_loader)
    writer = SummaryWriter(logdir=args.save_path)

    # if args.resume:
    #     # load checkpoint
    #     state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))
    #     init_epoch = state['epoch']
    #     resumed_state = state['state_dict']
    #     model.load_state_dict(resumed_state)
    #     trlog = state['trlog']
    #     optimizer.load_state_dict(state['optimizer'])
    #     global_count = state['global_count']
    # else:
    init_epoch = 1
    trlog = {}
    trlog['args'] = vars(args)
    trlog['train_loss'] = []
    trlog['val_loss_dist'] = []
    trlog['val_loss_sim'] = []
    trlog['train_acc'] = []
    trlog['val_acc_sim'] = []
    trlog['val_acc_dist'] = []
    trlog['max_acc_dist'] = 0.0
    trlog['max_acc_dist_epoch'] = 0
    trlog['max_acc_sim'] = 0.0
    trlog['max_acc_sim_epoch'] = 0
    global_count = 0

    # warmup
    for epoch in range(args.warmup_epochs):
        model = train(args, model, train_loader, criterion, optimizer, lr_schedule_warmup, None)
    if args.warmup_epochs > 0:
        validate(args, model, val_loader, -1, None)
    trlog = validate(args, model, val_loader, -1, trlog)
    for epoch in range(args.max_epoch):
        tic = time.time()
        model, trlog = train(args, model, train_loader, criterion, optimizer, lr_schedule, trlog)
        elapsed = time.time() - tic
        if epoch % args.eval_interval == 0:
            trlog = validate(args, model, val_loader, epoch, trlog)
        print(f'Epoch: {epoch}, Time cost: {elapsed}')
    save_model('epoch-last', model, args)
    writer.close()
    # import pdb
    #
    # pdb.set_trace()
