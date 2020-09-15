import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from model.utils import euclidean_metric
from ..utils import one_hot


# implement the SimpleShot

class SimpleShot(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res18':
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res12':
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        else:
            raise ValueError('')

    def forward(self, data, split_index, seen_meam, subtract_support=True, subtract_query=True):
        data = self.encoder(data)
        dim = data.shape[-1]
        shot, query = data[:split_index], data[split_index:]

        if subtract_support:
            shot = shot - seen_meam

        if subtract_query:
            query = query - seen_meam

        proto = shot.reshape(self.args.shot, -1, dim).mean(dim=0)
        proto = F.normalize(proto, dim=1, p=2)
        logits_sim = torch.mm(query, proto.t())
        return logits_sim
