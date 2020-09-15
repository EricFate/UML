import argparse
import numpy as np
import torch
from tqdm import tqdm
from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler, ClassSampler
from torch.utils.data import DataLoader
from model.utils import pprint, set_gpu, Averager, compute_confidence_interval
from sklearn.svm import LinearSVC
from warnings import filterwarnings
from model.utils import get_dataset
from eval_simpleshot import get_class_mean
import torch.nn.functional as F
import os.path as osp

filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12'])
    parser.add_argument('--datasets', type=str, nargs='+', default=['MiniImageNet', 'CUB'],
                        # choices=['MiniImageNet', 'CUB', 'TieredImageNet']
                        )
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--use_memory', action='store_true', default=True)
    parser.add_argument('--init_weights', type=str,
                        default='./FixRes-MiniImageNet-Res12-140LS0.0MX0.0/Bsz32Epoch-3-Cos-lr0.0008decay0.0001/max_acc_sim.pth')
    parser.add_argument('--unsupervised', action='store_true', default=False)
    parser.add_argument('--augment', type=str, default='none')
    parser.add_argument('--test_size', type=int, default=84)
    parser.add_argument('-e', '--num_test_episodes', type=int, default=10000)
    parser.add_argument('-c', '--centralize', action='store_true', default=False)
    parser.add_argument('-n', '--normalize', action='store_true', default=False)

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

    trainset = get_dataset('MiniImageNet', 'train', False, args)

    class_mean = get_class_mean('MiniImageNet', args.backbone_class, trainset)
    # testsets = dict(((n, get_dataset(n, 'test', args.unsupervised, args)) for n in args.eval_dataset.split(',')))
    ensemble_result = []
    for n in args.datasets:
        print('----------- test on {} --------------'.format(n))
        valset = get_dataset(n, 'val', args.unsupervised, args)
        for i, (args.way, args.shot) in enumerate(valset.eval_setting):
            # train best gamma
            # valset = Dataset('val', args.unsupervised, args)
            valset = get_dataset(n, 'val', args.unsupervised, args)
            val_sampler = CategoriesSampler(valset.label, 500, min(args.way, valset.num_class), args.shot + args.query)
            val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
            # test_set = Dataset('test', args.unsupervised, args)
            test_set = get_dataset(n, 'test', args.unsupervised, args)
            sampler = CategoriesSampler(test_set.label, args.num_test_episodes, args.way, args.shot + args.query)
            loader = DataLoader(dataset=test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
            shot_label = torch.arange(min(args.way, valset.num_class)).repeat(args.shot).numpy()
            query_label = torch.arange(min(args.way, valset.num_class)).repeat(args.query).numpy()
            val_acc_record = np.zeros((500, len(c_list)))

            ave_acc = Averager()

            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader, 1), total=len(val_loader), desc='val eval'):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data = batch[0]

                    data_emb = model(data, is_emb=True)
                    if args.centralize:
                        data_emb = data_emb - class_mean
                    if args.normalize:
                        data_emb = F.normalize(data_emb, dim=1, p=2)

                    split_index = min(args.way, valset.num_class) * args.shot
                    data_shot, data_query = data_emb[:split_index], data_emb[split_index:]

                    for j, c in enumerate(c_list, 0):
                        SVM = LinearSVC(C=c, multi_class='crammer_singer', dual=False, max_iter=5000).fit(
                            data_shot.cpu().numpy(), shot_label)
                        prediction = SVM.predict(data_query.cpu().numpy())
                        acc = np.mean(prediction == query_label)
                        val_acc_record[i - 1, j] = acc
                    # print('batch {}: {}'.format(i, ','.join(['{:.2f}'.format(e*100) for e in val_acc_record[i - 1, :]])))

            val_acc_record = np.mean(val_acc_record, 0)
            best_c = c_list[np.argmax(val_acc_record)]
            print(best_c)

            test_acc_record = np.zeros((args.num_test_episodes,))
            shot_label = torch.arange(args.way).repeat(args.shot).numpy()
            query_label = torch.arange(args.way).repeat(args.query).numpy()

            with torch.no_grad():
                for i, batch in tqdm(enumerate(loader, 1), total=len(loader), desc='test eval'):
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data = batch[0]
                    data_emb = model(data, is_emb=True)

                    if args.centralize:
                        data_emb = data_emb - class_mean
                    if args.normalize:
                        data_emb = F.normalize(data_emb, dim=1, p=2)

                    split_index = args.way * args.shot
                    data_shot, data_query = data_emb[:split_index], data_emb[split_index:]

                    SVM = LinearSVC(C=best_c, multi_class='crammer_singer', dual=False, max_iter=5000).fit(
                        data_shot.cpu().numpy(), shot_label)
                    prediction = SVM.predict(data_query.cpu().numpy())
                    acc = np.mean(prediction == query_label)
                    test_acc_record[i - 1] = acc
                    # print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

            m, pm = compute_confidence_interval(test_acc_record)
            ensemble_result.append('{:.4f} + {:.4f}'.format(m, pm))
            print('{} way {} shot,Test acc={:.4f} + {:.4f}, best_gamma:{}'.format(args.way, args.shot, m, pm, best_c))
    print('ensemble result: {}'.format(','.join(ensemble_result)))
