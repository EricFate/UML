import torch as t
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from model.utils import get_dataset
from PIL import Image
import numpy as np
import os.path as osp
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--way', type=int, default=5)
    parser.add_argument('-s', '--shot', type=int, default=1)
    parser.add_argument('-q', '--query', type=int, default=1)
    parser.add_argument('--test_size', type=int, default=84)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'CUB', 'TieredImageNet']
                        )
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--use_memory', action='store_true', default=True)
    parser.add_argument('--init_weights', type=str, default='./checkpoints/best/con_pre.pth')
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--augment', type=str, default='none')
    parser.add_argument('--filename', type=str, default='qsim_pairs')
    parser.add_argument('-e', '--num_test_episodes', type=int, default=100)

    args = parser.parse_args()
    args.num_class = 64
    args.orig_imsize = -1
    trainset = get_dataset(args.dataset, 'train', args.unsupervised, args)
    testset = get_dataset(args.dataset, 'test', args.unsupervised, args)

    pairs = t.load('%s.pt' % args.filename)[:args.num_test_episodes]
    tsne_data = t.load('tsne.pt')
    all_tsne = tsne_data['tsne_embs']
    train_tsne = tsne_data['tsne_embs'][:tsne_data['num_train']]
    test_tsne = tsne_data['tsne_embs'][tsne_data['num_train']:]
    for task, (support, query, neg, better_indexes, loss, acc) in tqdm(enumerate(pairs), total=len(pairs)):

        # --------------------- draw raw  ----------------------------
        num_col = args.shot + 1 + args.query + 1 + args.query
        fig = plt.figure()
        for i, idx in enumerate(support):
            data = testset.data[idx]
            im = Image.open(data).convert('RGB')
            ax = plt.subplot(args.way, num_col, i * num_col + 1)
            plt.imshow(im)
            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            # 去除黑框
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)

        query_start = args.shot + 1
        for (i, j), idx in np.ndenumerate(query.numpy()):
            data = testset.data[idx]
            im = Image.open(data).convert('RGB')
            ax = plt.subplot(args.way, num_col, j * num_col + query_start + i + 1)
            plt.imshow(im)
            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])
            # 去除黑框
            # ax.spines['top'].set_visible(False)
            # ax.spines['right'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            # ax.spines['left'].set_visible(False)
        neg_start = args.shot + 1 + args.query + 1
        for (i, j), idx in np.ndenumerate(neg.numpy()):
            data = trainset.data[idx]
            im = Image.open(data).convert('RGB')
            ax = plt.subplot(args.way, num_col, j * num_col + neg_start + i + 1)
            plt.imshow(im)
            # plt.axis('off')
            plt.xticks([])
            plt.yticks([])

        # 设置各个子图间间距
        plt.subplots_adjust(wspace=0.02, hspace=0.02)
        plt.suptitle('loss : %.4f  acc : %.4f' % (loss, acc))
        root = osp.join('fig', args.filename)
        raw_dir = osp.join(root, 'raw')
        if not osp.exists(raw_dir):
            os.makedirs(raw_dir)
        plt.savefig(osp.join(raw_dir, '%s.png' % task))
        plt.close(fig)

        # --------------------- draw tsne  ----------------------------
        fig = plt.figure()
        plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c='lightcoral')
        plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c='skyblue')
        support_tsne = test_tsne[support.numpy()]
        query_tsne = test_tsne[query.numpy()]
        # neg_tsne = train_tsne[neg.numpy().squeeze()]
        markers = ['.', '^', '+', '*', 'x']
        for i in range(args.way):
            plt.scatter(support_tsne[i, 0], support_tsne[i, 1], c='r', marker=markers[i])
            plt.scatter(query_tsne[:, i, 0], query_tsne[:, i, 1], c='g', marker=markers[i])
            for j in range(args.query):
                neg_tsne = train_tsne[better_indexes[j][i]]
                plt.scatter(neg_tsne[:, 0], neg_tsne[:, 1], c='b', marker=markers[i])

        tsne_dir = osp.join(root, 'tsne')
        if not osp.exists(tsne_dir):
            os.makedirs(tsne_dir)
        plt.savefig(osp.join(tsne_dir, '%s.png' % task))
        plt.close(fig)
