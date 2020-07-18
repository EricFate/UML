import torch
import numpy as np
import torch.nn.functional as F
from model.models import FewShotModelWrapper, FewShotModel
from torch import nn


class TaskGateWrapper(FewShotModelWrapper):

    def __init__(self, args, model: FewShotModel):
        super().__init__(args, model)
        # projection MLP
        num_ftrs = self.hdim
        self.gate = nn.Sequential(nn.Linear(num_ftrs, num_ftrs),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(num_ftrs, num_ftrs))

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        task_idx = torch.cat((support_idx, query_idx), dim=1)
        # (num_tasks, num_shot + num_query, num_way, num_feature)
        task = instance_embs[task_idx]
        # (num_tasks, num_shot + num_query, num_way, num_feature)
        task_gate = self.gate(task)
        # (num_tasks, 1, 1, num_feature)
        task_gate = torch.sum(task_gate, dim=(1, 2), keepdim=True)
        # (num_tasks, 1, 1, num_feature)
        task_gate = torch.sigmoid(task_gate)

        task_features = task * task_gate + task

        # # organize support/query data
        support, query = torch.split(task_features, [support_idx.size(1),
                                                     query_idx.size(1)], dim=1)

        return self.model._forward_task(emb_dim, support, query)

    def load_state_dict(self, state_dict, strict=True):
        # nn.Module.load_state_dict(self, state_dict, strict)
        self.model.load_state_dict(state_dict['model'], strict)
        self.gate.load_state_dict(state_dict['gate'], strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # return nn.Module.state_dict(self, destination, prefix, keep_vars)
        return {'model': self.model.state_dict(destination, prefix, keep_vars),
                'gate': self.gate.state_dict(destination, prefix, keep_vars)}
