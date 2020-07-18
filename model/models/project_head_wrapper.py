import torch
import numpy as np
import torch.nn.functional as F
from model.models import FewShotModelWrapper, FewShotModel
from torch import nn


class ProjectionHeadWrapper(FewShotModelWrapper):

    def __init__(self, args, model: FewShotModel):
        super().__init__(args, model)
        # projection MLP
        num_ftrs = self.hdim
        out_dim = int(self.hdim * args.hidden_ratio)
        self.projection = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(num_ftrs, out_dim))

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                instance_embs = self.projection(instance_embs)
                logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        return self.model._forward(x, support_idx, query_idx)
