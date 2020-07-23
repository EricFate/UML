import os
import shutil
import time
import pprint
import torch
import argparse
import numpy as np
import os
from tensorboardX import SummaryWriter


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


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.

    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    if indices.is_cuda:
        encoded_indicies = encoded_indicies.cuda()
    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(dir_path, scripts_to_save=None):
    if os.path.exists(dir_path):
        if input('{} exists, remove? ([y]/n)'.format(dir_path)) != 'n':
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
    else:
        os.mkdir(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for src_file in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(src_file))
            print('copy {} to {}'.format(src_file, dst_file))
            if os.path.isdir(src_file):
                shutil.copytree(src_file, dst_file)
            else:
                shutil.copyfile(src_file, dst_file)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def postprocess_args(args):
    save_path1 = '-'.join([args.dataset, args.eval_dataset, args.model_class, args.backbone_class,
                           '{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query), args.additional])
    save_path2 = '_'.join([str('_'.join(args.step_size.split(','))), str(args.gamma),
                           'lr{:.2g}mul{:.2g}'.format(args.lr, args.lr_mul),
                           str(args.lr_scheduler),
                           'T1{}T2{}'.format(args.temperature, args.temperature2),
                           'b{}'.format(args.balance),
                           'bsz{:03d}'.format(max(args.way, args.num_classes) * (args.shot + args.query)),
                           'batch{:03d}'.format(args.batch_size),
                           'ntask{:03d}'.format(args.num_tasks),
                           'nclass{:03d}'.format(args.num_classes),
                           'ep{}'.format(args.max_epoch),
                           'eval{}'.format(args.eval)
                           ])

    if args.unsupervised:
        save_path1 = 'taco' + save_path1

    if args.init_weights is not None:
        save_path1 += '-Pre_%s' % (args.init_weights.strip().split('/')[-1].split('.')[0])
    if args.use_euclidean:
        save_path1 += '-DIS'
    else:
        save_path1 += '-SIM'

    if args.additional == 'ProjectionHead':
        save_path1 += '_hr%.1f' % args.hidden_ratio

    if args.fix_BN:
        save_path2 += '-FBN'
    save_path2 += '-Aug_%s' % args.augment
    if args.additional == 'MixUp':
        save_path2 = 'al_%.1f_layer_%d_rand_%s' % (args.alpha, args.layer, args.rand) + save_path2

    if args.model_class == 'DummyProto':
        save_path2 = 'nd_%d_sp_%d' % (args.dummy_nodes, args.dummy_samples) + save_path2

    if args.model_class == 'MemoryBankProto':
        save_path2 = 'K%d_frac%.4f_m%.4f_p_%s_s_%d_sp_%s_' % (args.K, args.bank_ratio, args.m, args.max_pool,args.start,args.split) + save_path2

    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    args.save_path = os.path.join(args.save_dir, save_path1, save_path2)
    args.filename = os.path.join(save_path1, save_path2)
    return args


def get_command_line_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--episodes_per_epoch', type=int, default=100)
    parser.add_argument('--num_eval_episodes', type=int, default=600)
    parser.add_argument('--model_class', type=str, default='ProtoNet',
                        choices=['MatchNet', 'ProtoNet', 'BILSTM', 'DeepSet', 'GCN', 'FEAT',
                                 'FEATSTAR', 'DummyProto', 'ExtremeProto',
                                 'MemoryBankProto'])  # None for MatchNet or ProtoNet
    parser.add_argument('--use_euclidean', action='store_true', default=False)
    parser.add_argument('--backbone_class', type=str, default='WRN',
                        choices=['ConvNet', 'Res12', 'Res18', 'WRN'])

    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'TieredImageNet', 'CUB'])
    parser.add_argument('--eval_dataset', type=str, default='MiniImageNet,CUB')
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--additional', type=str, default='none')
    # choices=['none', 'TaskContrastive', 'MixUp', 'Mixed', 'ProjectionHead','TaskGate'])
    parser.add_argument('--eval_all', action='store_true', default=False)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--dummy_nodes', type=int, default=5)
    parser.add_argument('--dummy_samples', type=int, default=15)
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--eval_way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--eval_shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--eval_query', type=int, default=15)
    parser.add_argument('--balance', type=float, default=1)
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--temperature2', type=float, default=1)  # the temperature in the

    # param for mixup
    parser.add_argument('--alpha', type=float, default=0.2)  # beta distribution parameter
    parser.add_argument('--layer', type=int, default=3)  # beta distribution parameter
    parser.add_argument('--rand', action='store_true', default=False)

    # param for memory bank
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--bank_ratio', type=float, default=0.)
    parser.add_argument('--m', type=float, default=0.999)
    parser.add_argument('--max_pool', action='store_true', default=False)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--split', action='store_true', default=False)

    # param for projection head
    parser.add_argument('--hidden_ratio', type=float, default=1.)

    # optimization parameters
    parser.add_argument('--orig_imsize', type=int,
                        default=-1)  # -1 for no cache, and -2 for no resize, only for MiniImageNet and CUB
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_mul', type=float, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='step', choices=['multistep', 'step', 'cosine'])
    parser.add_argument('--step_size', type=str, default='20')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--fix_BN', action='store_true',
                        default=False)  # means we do not update the running mean/var in BN, not to freeze BN
    parser.add_argument('--augment', type=str, default='none')
    parser.add_argument('--multi_gpu', action='store_true', default=False)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--init_weights', type=str, default=None)

    # usually untouched parameters
    parser.add_argument('--mom', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)  # we find this weight decay value works the best
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--path', type=str, default='')
    return parser


writer = None


def init_summary_writer(name):
    global writer
    writer = SummaryWriter(log_dir='runs/%s' % name)


def get_summary_writer():
    global writer
    return writer


def get_dataset(dataset, setname, unsupervised, args, augment='none'):
    if dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif dataset == 'TieredImageNet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    else:
        raise ValueError('Non-supported Dataset.')
    return Dataset(setname, unsupervised, args, augment)
