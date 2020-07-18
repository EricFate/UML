import time
import os.path as osp
import numpy as np

import torch
import torch.nn.functional as F

from model.trainer.base import Trainer
from model.trainer.helpers import (
    get_dataloader, prepare_model, prepare_optimizer,
)
from model.utils import (
    pprint, ensure_path,
    Averager, Timer, count_acc, one_hot,
    compute_confidence_interval,
)
from tensorboardX import SummaryWriter
from collections import deque
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.dataloader.samplers import CategoriesSampler
from model.utils import init_summary_writer


# inter_grad = {}
#
#
# def save_grad(name):
#     def hook(grad):
#         inter_grad[name] = grad
#
#     return hook


class FSLTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)

        self.train_loader, self.valset, self.testset = get_dataloader(args)
        init_summary_writer(args.filename)
        self.model, self.para_model = prepare_model(args)
        # for n, p in self.para_model.named_parameters():
        #     p.register_hook(save_grad(n))
        self.optimizer, self.lr_scheduler = prepare_optimizer(self.model, args)

    def prepare_label(self):
        args = self.args

        # prepare one-hot label
        label = torch.arange(args.way, dtype=torch.int16).repeat(
            args.num_tasks * args.query  # *(self.train_loader.num_device if args.multi_gpu else 1)
        )
        label_aux = torch.arange(args.way, dtype=torch.int8).repeat(
            args.num_tasks * (args.shot + args.query)  # *(self.train_loader.num_device if args.multi_gpu else 1)
        )

        label = label.type(torch.LongTensor)
        label_aux = label_aux.type(torch.LongTensor)

        if torch.cuda.is_available():
            label = label.cuda()
            label_aux = label_aux.cuda()

        return label, label_aux

    # @profile
    def train(self):
        args = self.args
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()

        # start FSL training
        label, label_aux = self.prepare_label()
        for epoch in range(1, args.max_epoch + 1):
            self.train_epoch += 1
            self.model.train()
            if self.args.fix_BN:
                self.model.encoder.eval()
            self.model.set_epoch(epoch)
            # tl1 = Averager()
            # tl2 = Averager()
            # ta = Averager()

            start_tm = time.time()
            for batch in tqdm(self.train_loader, total=len(self.train_loader), desc='train epoch %d' % epoch):
                self.train_step += 1

                if torch.cuda.is_available():
                    data, data_ = [_.cuda() for _ in batch]
                else:
                    data, data_ = batch[0], batch[1]

                data_tm = time.time()
                self.dt.add(data_tm - start_tm)

                # get saved centers
                if self.args.additional == 'Mixed' or args.model_class == 'DummyProto':
                    logits, reg_loss = self.para_model((data, data_))
                else:
                    logits, reg_loss = self.para_model(data)
                if reg_loss is not None:
                    if logits is None:
                        loss = 0
                    else:
                        loss = F.cross_entropy(logits, label)
                    total_loss = loss + args.balance * reg_loss
                else:
                    loss = F.cross_entropy(logits, label)
                    total_loss = F.cross_entropy(logits, label)

                # tl2.add(loss)
                forward_tm = time.time()
                self.ft.add(forward_tm - data_tm)
                # acc = count_acc(logits, label)

                # tl1.add(total_loss.item())
                # ta.add(acc)

                self.optimizer.zero_grad()
                total_loss.backward()
                backward_tm = time.time()
                self.bt.add(backward_tm - forward_tm)

                self.optimizer.step()
                optimizer_tm = time.time()
                self.ot.add(optimizer_tm - backward_tm)

                # refresh start_tm
                start_tm = time.time()

            # if args.model_class == 'DummyProto':
            #     print('dummy count: %s' % str(self.para_model.activate_counter))
            self.lr_scheduler.step()
            self.try_evaluate(epoch)

            print('ETA:{}/{}'.format(
                self.timer.measure(),
                self.timer.measure(self.train_epoch / args.max_epoch))
            )

        torch.save(self.trlog, osp.join(args.save_path, 'trlog'))
        self.save_model('epoch-last')

    def evaluate(self, data_loader):
        # restore model args
        args = self.args
        # evaluation mode
        self.model.eval()
        record = np.zeros((args.num_eval_episodes, 2))  # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(
            # args.num_tasks *
            args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(data_loader, 1), total=len(data_loader), desc='eval procedure'):
                if torch.cuda.is_available():
                    data, _ = [_.cuda() for _ in batch]
                else:
                    data = batch[0]

                logits = self.model(data)
                loss = F.cross_entropy(logits, label)
                acc = count_acc(logits, label)
                record[i - 1, 0] = loss.item()
                record[i - 1, 1] = acc

        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])

        # train mode
        self.model.train()
        if self.args.fix_BN:
            self.model.encoder.eval()
        return vl, va, vap

    def evaluate_test(self, **kwargs):
        # restore model args
        args = self.args
        # evaluation mode
        if 'path' in kwargs:
            path = kwargs['path']
        else:
            path = osp.join(self.args.save_path, 'max_acc.pth')
        print('model path %s' % path)
        model_dict = self.model.state_dict()
        if args.augment == 'moco':
            pretrained_dict = torch.load(path)['state_dict']
            prefix = 'module.encoder_q'
            pretrained_dict = {'encoder'+k[len(prefix):]: v for k, v in pretrained_dict.items() if k.startswith(prefix)}
        else:
            pretrained_dict = torch.load(path)['params']
            # if args.backbone_class == 'ConvNet':
            #     pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # pretrained_dict = torch.load(path)['params']
        # if args.backbone_class == 'ConvNet':
        #     pretrained_dict = {'encoder.' + k: v for k, v in pretrained_dict.items()}
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(pretrained_dict.keys())
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.model.eval()
        print('best epoch {}, best val acc={:.4f} + {:.4f}'.format(
            self.trlog['max_acc_epoch'],
            self.trlog['max_acc'],
            self.trlog['max_acc_interval']))
        with torch.no_grad():
            with open(osp.join(self.args.save_path,
                               'test_result%s' % ('_eval_all' if args.eval_all else '')),
                      'w') as f:
                for d, testset in self.testset.items():
                    print('----------- test on {} --------------'.format(d))
                    f.write('----------- test on {} --------------\n'.format(d))
                    # test_sampler = CategoriesSampler(testset.label,
                    #                                  10000,  # args.num_eval_episodes,
                    #                                  args.eval_way, args.eval_shot + args.eval_query)
                    # test_loader = DataLoader(dataset=testset,
                    #                          batch_sampler=test_sampler,
                    #                          num_workers=args.num_workers,
                    #                          pin_memory=True)
                    if args.eval_all:
                        for args.eval_way, args.eval_shot in testset.eval_setting:
                            vl, va, vap = self.test_process(testset)
                            f.write('{} way {} shot,Test acc={:.4f} + {:.4f}\n'.format(args.eval_way, args.eval_shot,
                                                                                       va,
                                                                                       vap))
                    else:
                        vl, va, vap = self.test_process(testset)
                        f.write('{} way {} shot,Test acc={:.4f} + {:.4f}\n'.format(args.eval_way, args.eval_shot,
                                                                                   va,
                                                                                   vap))

    def test_process(self, testset):
        args = self.args
        record = np.zeros((10000, 2))  # loss and acc
        label = torch.arange(args.eval_way, dtype=torch.int16).repeat(
            # args.num_tasks *
            args.eval_query)
        label = label.type(torch.LongTensor)
        if torch.cuda.is_available():
            label = label.cuda()
        test_sampler = CategoriesSampler(testset.label,
                                         10000,  # args.num_eval_episodes,
                                         args.eval_way, args.eval_shot + args.eval_query)
        test_loader = DataLoader(dataset=testset,
                                 batch_sampler=test_sampler,
                                 num_workers=args.num_workers,
                                 pin_memory=True)
        for i, batch in tqdm(enumerate(test_loader, 1), total=len(test_loader)):
            if torch.cuda.is_available():
                data, _ = [_.cuda() for _ in batch]
            else:
                data = batch[0]

            logits = self.model(data)
            loss = F.cross_entropy(logits, label)
            acc = count_acc(logits, label)
            record[i - 1, 0] = loss.item()
            record[i - 1, 1] = acc
        assert (i == record.shape[0])
        vl, _ = compute_confidence_interval(record[:, 0])
        va, vap = compute_confidence_interval(record[:, 1])
        print('{} way {} shot,Test acc={:.4f} + {:.4f}\n'.format(args.eval_way, args.eval_shot,
                                                                 va,
                                                                 vap))
        return vl, va, vap

    def final_record(self):
        # save the best performance in a txt file
        pass
