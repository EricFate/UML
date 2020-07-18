from collections import OrderedDict
from typing import Union, Dict, overload

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch import Tensor

from model.models import FewShotModelWrapper


class TaskContrastiveWrapper(FewShotModelWrapper):

    def _forward(self, instance_embs: torch.Tensor, support_idx, query_idx):
        if not self.training:
            logits = self.model._forward(instance_embs, support_idx, query_idx)
            return logits
        logits, _ = self.model._forward(instance_embs, support_idx, query_idx)
        contrastive_losses = []
        for i in range(128):
            sampled = np.random.randint(0, instance_embs.size(0))
            cls = sampled % self.args.num_classes
            task_idx = torch.cat([support_idx, query_idx], dim=1).view(self.args.num_tasks, -1)
            tasks = torch.any((task_idx % self.args.num_tasks) == cls, dim=1)
            task_embed = instance_embs[task_idx]
            ins_embed = instance_embs[sampled].expand_as(task_embed)
            if self.args.use_euclidean:
                dists = - torch.mean(torch.sum((task_embed - ins_embed) ** 2, dim=-1), dim=-1) / self.args.temperature
            else:
                dists = torch.mean(
                    torch.sum(F.normalize(ins_embed, 2, dim=-1) * F.normalize(task_embed, 2, dim=-1), dim=-1),
                    dim=-1) / self.args.temperature
            num_pos = torch.sum(tasks).item()
            pseudo_label = torch.arange(self.args.num_tasks)[tasks].to(self.args.device)
            clogits = dists.repeat(num_pos, 1)
            # num_pos = torch.sum(tasks).item()
            # pos_dist = dists[tasks].unsqueeze(1)
            # neg_dist = dists[~tasks]
            # clogits = torch.cat((pos_dist, neg_dist.repeat(num_pos, 1)), dim=1)
            # pseudo_label = torch.zeros(num_pos).long().to(self.args.device)
            contrastive_losses.append(F.cross_entropy(clogits, pseudo_label))
        contrastive_loss = sum(contrastive_losses) / 128
        return logits, contrastive_loss
