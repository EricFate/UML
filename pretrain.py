import argparse
import os
import os.path as osp
import shutil
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
# from model.nn.loss import LabelSmoothing, NLLMultiLabelSmooth
# from model.utils import pprint, set_gpu, mkdir, Averager, Timer, count_acc, euclidean_metric, MixUpWrapper
from model.utils import pprint, set_gpu, Averager, Timer, count_acc, euclidean_metric


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImagenet'])
    parser.add_argument('--backbone_class', type=str, default='ConvNet', choices=['ConvNet', 'ConvAttNet', 'Res12',
                                                                                  'Res18', 'ResNest12', 'SAN'])
    parser.add_argument('--query', type=int, default=15)

    parser.add_argument('--label-smoothing', type=float, default=0.0, help='label-smoothing (default eta: 0.0)')
    parser.add_argument('--mixup', type=float, default=0.0, help='mixup (default eta: 0.0)')
    parser.add_argument('--rand-aug', action='store_true', default=False, help='random augment')
    parser.add_argument('--augment', type=str, default='none')

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='multistep', help='learning rate scheduler (default: cos)')
    parser.add_argument('--step', type=str, default='[2,3,4,5]', help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--no-bn-wd', action='store_true', default=False, help='no bias decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='number of warmup epochs (default: 0)')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='SGD weight decay (default: 1e-4)')

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--finetune', action='store_true', default=False)
    parser.add_argument('--samples_per_class', type=int, default=50)

    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    args.orig_imsize = -1
    pprint(vars(args))
    set_gpu(args.gpu)

    save_path1 = '-'.join(['PreTrain', args.dataset, args.backbone_class])
    save_path1 += 'LS{}MX{}'.format(args.label_smoothing, args.mixup)
    if args.lr_scheduler == 'cosine':
        save_path2 = 'Bsz{}Epoch-{}-Cos-lr{}decay{}'.format(args.batch_size, args.max_epoch, args.lr, args.weight_decay)
    elif args.lr_scheduler == 'step':
        save_path2 = 'Bsz{}Epoch-{}-Step-lr{}-{}-{}decay{}'.format(args.batch_size, args.max_epoch, args.lr, args.step,
                                                                   args.gamma, args.weight_decay)
    elif args.lr_scheduler == 'multistep':
        save_path2 = 'Bsz{}Epoch-{}-MultiStep-lr{}-{}-{}decay{}'.format(args.batch_size, args.max_epoch, args.lr,
                                                                        args.step, args.gamma, args.weight_decay)
    else:
        raise ValueError('No Such Schedule')

    if args.warmup_epochs > 0:
        save_path2 += 'Warmup{}'.format(args.warmup_epochs)

    if args.rand_aug:
        save_path2 += '-RandAug'

    if args.no_bn_wd:
        save_path2 += '-no-bn-wd'

    args.save_path = osp.join(save_path1, save_path2)
    if not os.path.exists(os.path.join(args.save_dir, save_path1)):
        os.mkdir(os.path.join(args.save_dir, save_path1))
    return args


def get_loader(args):
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImagenet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset
    else:
        raise ValueError('Non-supported Dataset.')

    if args.finetune:
        split = 'train_%d' % args.samples_per_class
    else:
        split = 'train'

    trainset = Dataset(split, False, args, augment=args.augment)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                              pin_memory=True)
    args.num_class = trainset.num_class
    valset = Dataset('val', False, args)
    args.num_val_class = valset.num_class
    val_sampler = CategoriesSampler(valset.label, 200, valset.num_class, 1 + args.query)  # test on 16-way 1-shot
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)
    args.way = valset.num_class
    args.shot = 1
    return train_loader, val_loader


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


def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
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
            save_checkpoint(True)

        if va_sim > trlog['max_acc_sim']:
            trlog['max_acc_sim'] = va_sim
            trlog['max_acc_sim_epoch'] = epoch
            save_model('max_acc_sim', model, args)
            save_checkpoint(True)

        trlog['val_loss_dist'].append(vl_dist)
        trlog['val_acc_dist'].append(va_dist)
        trlog['val_loss_sim'].append(vl_sim)
        trlog['val_acc_sim'].append(va_sim)
        return trlog


if __name__ == '__main__':
    args = get_args()
    train_loader, val_loader = get_loader(args)
    model = Classifier(args)
    # if args.mixup > 0:
    #     train_loader = MixUpWrapper(args.mixup, args.num_class, train_loader)
    #     criterion = NLLMultiLabelSmooth(args.label_smoothing)
    # elif args.label_smoothing > 0.0:
    #     criterion = LabelSmoothing(args.label_smoothing)
    # else:
    criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()

    optimizer, lr_schedule, lr_schedule_warmup = get_optimizer(args, model, train_loader)
    writer = SummaryWriter(logdir=args.save_path)

    if args.resume:
        # load checkpoint
        state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        model.load_state_dict(resumed_state)
        trlog = state['trlog']
        optimizer.load_state_dict(state['optimizer'])
        global_count = state['global_count']
    else:
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

    for epoch in range(args.max_epoch):
        tic = time.time()
        model, trlog = train(args, model, train_loader, criterion, optimizer, lr_schedule, trlog)
        if epoch % 5 == 0:
            trlog = validate(args, model, val_loader, epoch, trlog)
        elapsed = time.time() - tic
        print(f'Epoch: {epoch}, Time cost: {elapsed}')

    save_model('epoch-last', model, args)
    writer.close()
    import pdb

    pdb.set_trace()
