import argparse
import numpy as np
import torch
from tqdm import tqdm
from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader
from model.utils import pprint, set_gpu, Averager, compute_confidence_interval
from sklearn.svm import LinearSVC
from warnings import filterwarnings
from model.utils import get_dataset
from metric_learn import LMNN
from sklearn.externals import joblib
import os.path as osp
import os

# filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'CUB', 'TieredImageNet']
                        )
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--use_memory', action='store_true', default=True)
    parser.add_argument('--init_weights', type=str, default='./saves/initialization/tieredimagenet/Res12-pre.pth')
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--augment', type=str, default='none')
    parser.add_argument('--filename', type=str, default='conv')

    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)
    # if args.dataset == 'MiniImageNet':
    #     from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    # elif args.dataset == 'CUB':
    #     from model.dataloader.cub import CUB as Dataset
    # elif args.dataset == 'TieredImageNet':
    #     from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    # else:
    #     raise ValueError('Non-supported Dataset.')

    args.num_class = 64
    args.orig_imsize = -1
    model = Classifier(args)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    c_list = np.logspace(-10, 10, 21, base=2)

    # load pre-trained model (no FC weights)
    model_dict = model.state_dict()
    if args.init_weights is not None:
        pretrained_dict = torch.load(args.init_weights)['params']
        # remove weights for FC
        # pretrained_dict = {'encoder.'+k: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'fc' not in k}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    model.eval()

    trainset = get_dataset(args.dataset, 'train', args.unsupervised, args)
    loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers,
                        pin_memory=True)

    embs = []
    labels = []
    with torch.no_grad():
        for i, batch in tqdm(enumerate(loader, 1), total=len(loader), desc='embedding'):
            if torch.cuda.is_available():
                data, label = batch[0].cuda(), batch[1]
            else:
                data, label = batch

            data_emb = model.encoder(data)
            embs.append(data_emb)
            labels.append(label)
    embs = torch.cat(embs).cpu().numpy()
    labels = torch.cat(labels).numpy()
    lmnn = LMNN(verbose=True)
    print('fitting data....')
    lmnn.fit(embs, labels)
    print('fitting data finished.')
    directory = 'checkpoints/lmnn/'
    if not osp.exists(directory):
        os.makedirs(directory)
    joblib.dump(lmnn, osp.join(directory, '%s.pkl' % args.filename))
