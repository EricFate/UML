from abc import ABC

import torch
import torch.nn as nn


class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.hdim = 64
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            self.hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            self.hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            self.hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10,
                                       0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')
        self.ep = 0
        self.gep = 0
        self.lep = 0

    def split_instances_normal(self, num_tasks, num_shot, num_query, num_way, num_class=None):
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

    def split_instances(self, data):
        args = self.args
        if self.training:
            if args.unsupervised:
                return self.split_instances_normal(args.num_tasks, args.shot,
                                                   args.query, args.way, args.batch_size)
            return self.split_instances_normal(args.num_tasks, args.shot,
                                               args.query, args.way, args.num_classes)
        else:
            return self.split_instances_normal(1, args.eval_shot,
                                               args.eval_query, args.eval_way)

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        emb_dim = x.size(-1)

        # organize support/query data
        support = x[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = x[query_idx.flatten()].view(*(query_idx.shape + (-1,)))
        return self._forward_task(emb_dim, support, query)

    def _forward_task(self, emb_dim, support, query):
        pass

    def set_epoch(self, ep):
        self.gep = ep
        self.lep = 0


class FewShotModelWrapper(FewShotModel, ABC):
    def __init__(self, args, model: FewShotModel):
        super().__init__(args)
        self.model = model
        self.encoder = self.model.encoder

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.model.state_dict(destination, prefix, keep_vars)
