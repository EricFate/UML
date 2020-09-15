import argparse
import numpy as np
import torch
from tqdm import tqdm
from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
from torch.utils.data import DataLoader, Dataset
from model.utils import pprint, set_gpu, Averager, compute_confidence_interval
from sklearn.svm import LinearSVC
from warnings import filterwarnings
from torch.nn import functional as F
from model.utils import get_dataset
import joblib
import os.path as osp
import os
from model.dataloader.samplers import CategoriesSampler

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


# filterwarnings('ignore')

def minus(pq_sim, pn_sim, qn_sim):
    return pq_sim - pn_sim - qn_sim


def multiply(pq_sim, pn_sim, qn_sim):
    return pq_sim * (pn_sim + qn_sim) / 2


def divide(pq_sim, pn_sim, qn_sim):
    return pq_sim / ((pn_sim + qn_sim) / 2)


def split_instances_normal(num_tasks, num_shot, num_query, num_way, num_class=None):
    num_class = num_way if (num_class is None or num_class < num_way) else num_class

    permuted_ids = torch.zeros(num_tasks, num_shot + num_query, num_way).long()
    for i in range(num_tasks):
        # select class indices
        clsmap = torch.randperm(num_class)[:num_way]
        # ger permuted indices
        for j, clsid in enumerate(clsmap):
            permuted_ids[i, :, j].copy_(
                torch.randperm((num_shot + num_query)) * num_class + clsid
            )

    if torch.cuda.is_available():
        permuted_ids = permuted_ids.cuda()

    support_idx, query_idx = torch.split(permuted_ids, [num_shot, num_query], dim=1)
    return support_idx, query_idx


def normalize(arr: np.ndarray):
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / norms


class IndexedDataset(Dataset):

    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset

    def __getitem__(self, index: int):
        data, label = self.dataset[index]
        return index, data, label

    def __len__(self) -> int:
        return len(self.dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--way', type=int, default=5)
    parser.add_argument('-s', '--shot', type=int, default=1)
    parser.add_argument('-q', '--query', type=int, default=1)
    parser.add_argument('-m', '--qsim_method', type=str, default='divide')
    parser.add_argument('--test_size', type=int, default=84)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'Res12'])
    parser.add_argument('--dataset', type=str, default='MiniImageNet',
                        choices=['MiniImageNet', 'CUB', 'TieredImageNet']
                        )
    parser.add_argument('--gpu', default='1')
    parser.add_argument('--use_memory', action='store_true', default=True)
    parser.add_argument('--init_weights', type=str, default='./checkpoints/best/con_pre.pth')
    # parser.add_argument('--emb_path', type=str, default='./checkpoints/emb/.pt')
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--augment', type=str, default='none')
    parser.add_argument('--filename', type=str, default='conv')
    parser.add_argument('-e', '--num_test_episodes', type=int, default=10000)

    # parser.add_argument('-n', '--num_train', type=int, default=1000)

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
    args.emb_path = osp.join('./checkpoints/emb', 'train_%s.pt' % args.backbone_class)
    if args.emb_path is None or not osp.exists(args.emb_path):
        trainset = get_dataset(args.dataset, 'train', args.unsupervised, args)
        loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=True)
        embs = []
        labels = []
        with torch.no_grad():
            for i, batch in tqdm(enumerate(loader, 1), total=len(loader), desc='train embedding'):
                if torch.cuda.is_available():
                    data, label = batch[0].cuda(), batch[1]
                else:
                    data, label = batch

                data_emb = model.encoder(data)
                embs.append(data_emb)
                labels.append(label)
        train_embs = torch.cat(embs)
        train_labels = torch.cat(labels)
        train_embs = F.normalize(train_embs, dim=-1)
        torch.save({'embs': train_embs, 'labels': train_labels}, args.emb_path)
    else:
        print('loading train embs')
        tmp = torch.load(args.emb_path)
        train_embs = tmp['embs']
        train_labels = tmp['labels']

    testset = get_dataset(args.dataset, 'test', args.unsupervised, args)
    # loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=False,
    #                     num_workers=args.num_workers,
    #                     pin_memory=True)

    # embs = []
    # labels = []
    # with torch.no_grad():
    #     for i, batch in tqdm(enumerate(loader, 1), total=len(loader), desc='test embedding'):
    #         if torch.cuda.is_available():
    #             data, label = batch[0].cuda(), batch[1]
    #         else:
    #             data, label = batch
    #         data_emb = model.encoder(data)
    #         embs.append(data_emb)
    #         labels.append(label)
    # test_embs = torch.cat(embs).cpu()
    # test_labels = torch.cat(labels)
    # train_embs = train_embs.cuda()

    # # np.random.choice(train_embs.size(0),args.num_train)
    # # (num_train,num_train)
    # tt_sim = torch.einsum('ij,kj->ik', train_embs, test_embs)
    # # (num_train,num_test)
    # tn_sim = torch.einsum('ij,kj->ik', train_embs, test_embs)
    # tt_sim = tt_sim.unsqueeze(2)
    # ttn_sim = tn_sim.unsqueeze(1) + tn_sim.unsqueeze(0) / 2
    # ttn_sim = tt_sim / ttn_sim

    print('sampling task')
    num_negs = train_labels.size(0)
    label = torch.arange(args.way, dtype=torch.int16).repeat(
        args.query * num_negs).cuda().long()
    test_sampler = CategoriesSampler(testset.label, args.num_test_episodes, args.way, args.shot + args.query)
    indexed_test_set = IndexedDataset(testset)
    test_loader = DataLoader(dataset=indexed_test_set,
                             batch_sampler=test_sampler,
                             num_workers=0,
                             pin_memory=True)
    reduce_label = torch.arange(args.way, dtype=torch.int16).repeat(
        args.query).cuda().long().view(args.query, args.way)
    neg_indexes = torch.arange(num_negs).cuda()
    pairs = []
    losses = []
    accs = []
    mean_benefit = 0.0
    count = 0
    position_stats = torch.zeros(num_negs, dtype=torch.long).cuda()
    with torch.no_grad():
        for i, (index, data, _) in tqdm(enumerate(test_loader, 1), total=len(test_loader),
                                        desc='tasks'):
            data = data.cuda()
            embs = model.encoder(data)
            # embs = F.normalize(embs, dim=-1)
            support_idx, query_idx = split_instances_normal(1, args.shot,
                                                            args.query, args.way)
            # (shot,way,emb)
            support = embs[support_idx.squeeze(0)]
            # (query,way,emb)
            query = embs[query_idx.squeeze(0)]
            # (way,emb)
            proto = torch.mean(support, dim=0)

            query = F.normalize(query, dim=-1)
            proto = F.normalize(proto, dim=-1)
            # (q,1,w,w)
            pq_sim = torch.einsum('ijl,kl->ijk', query, proto).unsqueeze(1)
            # (1,n,1,w)
            pn_sim = torch.einsum('il,jl->ij', train_embs, proto).unsqueeze(0).unsqueeze(2)
            # (q,n,w,1)
            qn_sim = torch.einsum('ijl,kl->ikj', query, train_embs).unsqueeze(3)

            outputs_origin = F.softmax(pq_sim, dim=-1)
            # logits = pq_sim / ((pn_sim + qn_sim) / 2)
            logits = eval(args.qsim_method)(pq_sim, pn_sim, qn_sim)

            outputs = F.softmax(logits, dim=-1)
            select_label = reduce_label.view(args.query, 1, args.way, 1)
            truth_origin = outputs_origin.gather(3, select_label).squeeze(-1)
            truth = outputs.gather(3, select_label.expand((args.query, num_negs, args.way, 1))).squeeze(-1)
            # bool (q,n,w)

            benefit_count = torch.sum((truth > truth_origin).float(), dim=1)
            # print(benefit_count / num_negs)
            mean_benefit = (mean_benefit * count + torch.mean(benefit_count / num_negs)) / (count + 1)
            count += 1
            # print(mean_benefit)
            better, index_better = torch.topk(truth - truth_origin, k=1, dim=1)
            b_index_better = better > 0
            sorted_index = torch.argsort(qn_sim.squeeze(3), dim=1, descending=True)
            reversed_sorted_index = torch.argsort(sorted_index, dim=1)  # reverse mapping
            positions = reversed_sorted_index.gather(1, index_better)
            position_stats[positions.view(-1)] += 1
            better_indexes = [[None for w in range(args.way)] for q in range(args.query)]

            print(pq_sim.cpu().numpy()[0, 0, 0, :])
            print(logits.gather(1, index_better.unsqueeze(3).expand_as(pq_sim)).cpu().numpy()[0, 0, 0, :])

            for q in range(args.query):
                for w in range(args.way):
                    better_indexes[q][w] = index_better[q, :, w][b_index_better[q, :, w]].cpu().numpy()

            logits = logits.reshape(-1, args.way)
            loss = F.cross_entropy(logits, label, reduction='none')
            loss = loss.view(args.query, num_negs, args.way)

            # find optimal neg opt by loss
            opt_neg = loss.argmin(dim=1, keepdim=True)
            pred = torch.argmax(logits, dim=-1)
            opt_neg_pred = pred.view(args.query, num_negs, args.way).gather(1, opt_neg).squeeze()
            pred_label = opt_neg_pred == reduce_label
            opt_neg_acc = pred_label.float().mean()
            # == label.view(args.query, num_negs, args.way)).float().mean(dim=(0, 2))
            mean_loss = loss.min(dim=1)[0].mean()

            pre_loss = mean_loss.cpu()
            pre_acc = opt_neg_acc.cpu()
            losses.append(pre_loss)
            accs.append(pre_acc)
            pair = (
                index[support_idx.squeeze()], index[query_idx.squeeze(0)], opt_neg.squeeze(1).cpu(),
                better_indexes,
                pre_loss, pre_acc)
            pairs.append(pair)
    print('loss : %.4f acc : %.4f' % (np.mean(losses), np.mean(accs)))
    print('mean benefit %.6f' % mean_benefit.item())
    path = os.path.join('checkpoints', 'qsim-%s' % args.qsim_method,
                        '-'.join(['{:02d}w{:02d}s{:02}q'.format(args.way, args.shot, args.query), args.dataset,
                                  args.backbone_class]))
    os.makedirs(path, exist_ok=True)
    torch.save(pairs, os.path.join(path, 'qsim_pairs.pt'))

    plt.figure()
    plt.ylabel('frequency')
    plt.xlabel('similar rank')
    cpu__numpy = position_stats.cpu().numpy()
    plt.ylim(0, np.max(cpu__numpy))
    plt.bar(x=np.arange(num_negs), height=cpu__numpy)
    plt.plot(cpu__numpy)
    plt.savefig(os.path.join(path, 'position_stats.png'))
    np.save(os.path.join(path, 'position_stats.npy'), cpu__numpy)
    # loader = DataLoader(dataset=testset, batch_size=args.batch_size, shuffle=False,
    #                     num_workers=args.num_workers,
    #                     pin_memory=True)
    #
    # embs = []
    # labels = []
    # with torch.no_grad():
    #     for i, batch in tqdm(enumerate(loader, 1), total=len(loader), desc='train embedding'):
    #         if torch.cuda.is_available():
    #             data, label = batch[0].cuda(), batch[1]
    #         else:
    #             data, label = batch
    #
    #         data_emb = model.encoder(data)
    #         embs.append(data_emb)
    #         labels.append(label)
    # test_embs = torch.cat(embs).cpu()
    # test_labels = torch.cat(labels)
    #
    # all_embs = torch.cat((train_embs.cpu(), test_embs), dim=0).numpy()
    # from sklearn.manifold import TSNE
    #
    # tsne = TSNE()
    # embs_ = tsne.fit_transform(all_embs)
    # torch.save({'tsne_embs': embs_, 'num_train': train_embs.size(0)}, 'tsne.pt')
