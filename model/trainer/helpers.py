import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler, RandomSampler, ClassSampler
from model.models.protonet import ProtoNet
from model.models.matchnet import MatchNet
# from model.models.feat import FEAT
# from model.models.featstar import FEATSTAR
# from model.models.deepset import DeepSet
# from model.models.bilstm import BILSTM
# from model.models.graphnet import GCN
from model.models.dummy_proto import DummyProto
from model.models.extreme_proto import ExtremeProto
from model.models.task_contrastive_wrapper import TaskContrastiveWrapper
from model.models import *

from model.utils import get_dataset

from model.models import wrappers


class MultiGPUDataloader:
    def __init__(self, dataloader, num_device):
        self.dataloader = dataloader
        self.num_device = num_device

    def __len__(self):
        return len(self.dataloader) // self.num_device

    def __iter__(self):
        data_iter = iter(self.dataloader)
        done = False

        while not done:
            try:
                output_batch = ([], [])
                for _ in range(self.num_device):
                    batch = next(data_iter)
                    for i, v in enumerate(batch):
                        output_batch[i].append(v[None])

                yield (torch.cat(_, dim=0) for _ in output_batch)
            except StopIteration:
                done = True
        return


def examplar_collate(batch):
    X, Y = [], []
    for b in batch:
        X.append(torch.stack(b[0]))
        Y.append(b[1])
    X = torch.stack(X)
    label = torch.LongTensor(Y)
    img = torch.cat(tuple(X.permute(1, 0, 2, 3, 4)), dim=0)
    return img, label


def get_dataloader(args):
    num_device = torch.cuda.device_count()
    num_episodes = args.episodes_per_epoch * num_device if args.multi_gpu else args.episodes_per_epoch
    num_workers = args.num_workers * num_device if args.multi_gpu else args.num_workers
    if args.additional == 'Mixed':
        from model.dataloader.mix_dataset import MixedDatasetWrapper
        trainset = get_dataset(args.dataset, 'train', True, args, augment=args.augment)
        # args.num_class = unsupervised_trainset.num_class
        unsupervised_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                                         num_workers=0,
                                         collate_fn=examplar_collate,
                                         pin_memory=True, drop_last=True)
        supervised_trainset = get_dataset(args.dataset, 'train', False, args, augment=args.augment)
        train_sampler = CategoriesSampler(supervised_trainset.label,
                                          num_episodes,
                                          max(args.way, args.num_classes),
                                          args.shot + args.query)

        supervised_loader = DataLoader(dataset=supervised_trainset,
                                       num_workers=num_workers,
                                       batch_sampler=train_sampler,
                                       pin_memory=True)
        dataset = MixedDatasetWrapper(supervised_loader, unsupervised_loader)
        train_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0,
                                  pin_memory=True)
    else:
        if args.finetune:
            split = 'train_%d' % args.samples_per_class
        else:
            split = 'train'
        trainset = get_dataset(args.dataset, split, args.unsupervised, args, augment=args.augment)
        if args.unsupervised:
            train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                      collate_fn=examplar_collate,
                                      pin_memory=True, drop_last=True)
        else:
            train_sampler = CategoriesSampler(trainset.label,
                                              num_episodes,
                                              max(args.way, args.num_classes),
                                              args.shot + args.query)

            train_loader = DataLoader(dataset=trainset,
                                      num_workers=num_workers,
                                      batch_sampler=train_sampler,
                                      pin_memory=True)
    if args.model_class == 'DummyProto':
        from model.dataloader.dummy_loader import DummyWrapper
        train_loader = DummyWrapper(args.dummy_samples, train_loader)
    # if args.multi_gpu and num_device > 1:
    # train_loader = MultiGPUDataloader(train_loader, num_device)
    # args.way = args.way * num_device

    valset = get_dataset(args.dataset, 'val', args.unsupervised, args)
    # val_sampler = CategoriesSampler(valset.label,
    #                                 args.num_eval_episodes,
    #                                 args.eval_way, args.eval_shot + args.eval_query)
    # val_loader = DataLoader(dataset=valset,
    #                         batch_sampler=val_sampler,
    #                         num_workers=args.num_workers,
    #                         pin_memory=True)
    #
    testsets = dict(((n, get_dataset(n, 'test', args.unsupervised, args)) for n in args.eval_dataset.split(',')))
    # testsets = TestDataset('test', args.unsupervised, args)
    # test_sampler = CategoriesSampler(testset.label,
    #                                  10000,  # args.num_eval_episodes,
    #                                  args.eval_way, args.eval_shot + args.eval_query)
    # test_loader = DataLoader(dataset=testset,
    #                          batch_sampler=test_sampler,
    #                          num_workers=args.num_workers,
    #                          pin_memory=True)
    args.image_shape = trainset.image_shape
    return train_loader, valset, testsets


def prepare_model(args):
    args.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = eval(args.model_class)(args)

    # load pre-trained model (no FC weights)
    if args.init_weights is not None:
        model_dict = model.state_dict()
        if args.augment == 'moco':
            pretrained_dict = torch.load(args.init_weights)['state_dict']
            pretrained_dict = {'encoder' + k[len('encoder_q'):]: v for k, v in pretrained_dict.items() if
                               k.startswith('encoder_q')}
        else:
            pretrained_dict = torch.load(args.init_weights)['params']
            # if args.backbone_class == 'ConvNet':
            #     pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    if args.additional != 'none':
        model = getattr(wrappers, args.additional + 'Wrapper')(args, model)
        # model = TaskContrastiveWrapper(args, model)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    model = model.to(device)
    if args.multi_gpu:
        model.encoder = nn.DataParallel(model.encoder, dim=0)
        para_model = model.to(device)
    else:
        para_model = model.to(device)
    # print(model.state_dict().keys())
    return model, para_model


def prepare_optimizer(model, args):
    top_para = [v for k, v in model.named_parameters() if 'encoder' not in k]
    # as in the literature, we use ADAM for ConvNet and SGD for other backbones
    if args.backbone_class == 'ConvNet':
        optimizer = optim.Adam(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            # weight_decay=args.weight_decay, do not use weight_decay here
        )
    else:
        optimizer = optim.SGD(
            [{'params': model.encoder.parameters()},
             {'params': top_para, 'lr': args.lr * args.lr_mul}],
            lr=args.lr,
            momentum=args.mom,
            nesterov=True,
            weight_decay=args.weight_decay
        )

    if args.lr_scheduler == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(args.step_size),
            gamma=args.gamma
        )
    elif args.lr_scheduler == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(_) for _ in args.step_size.split(',')],
            gamma=args.gamma,
        )
    elif args.lr_scheduler == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            args.max_epoch,
            eta_min=0  # a tuning parameter
        )
    else:
        raise ValueError('No Such Scheduler')

    return optimizer, lr_scheduler
