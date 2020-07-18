import torch
import numpy as np
import torch.nn.functional as F
from model.models import FewShotModelWrapper


class MixUpWrapper(FewShotModelWrapper):
    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        if not self.training:
            return self.model.forward(x)
        # split support query set for few-shot data

        logits, _ = self.model(x)

        support_idx, query_idx = self.split_instances(x)
        alpha = self.args.alpha
        c = np.random.beta(alpha, alpha)
        # c = 1.
        perm = torch.randperm(support_idx.size(0)).to(self.args.device)
        # mixed_x : (num_task,num_shot,num_way) + shape of image

        if self.args.rand:
            l = np.random.randint(len(self.encoder) + 1)
        else:
            l = self.args.layer
        pre_emb = self.encoder.pre_forward(x, l)

        mixed_emb = c * pre_emb[support_idx] + (1 - c) * pre_emb[support_idx[perm]]
        # find all queries
        q_idx, query_idx_ = torch.unique(query_idx, return_inverse=True)
        # q_x = x[q_idx]
        q_pre_emb = pre_emb[q_idx]
        # shape : (num_task*num_shot*num_way) + shape of image
        num_support = support_idx.size().numel()
        shape = (num_support,) + pre_emb.shape[1:]
        all_pre_emb = torch.cat([mixed_emb.view(*shape), q_pre_emb])
        instance_embs = self.encoder.post_forward(all_pre_emb, l)

        # compute support index
        support_idx_ = torch.arange(0, num_support).view(support_idx.size()).to(self.args.device)

        # transform query index
        query_idx_ += num_support
        # q_idx = list(q_idx.cpu().detach())
        # query_idx_ = query_idx.cpu().numpy()
        # transfrom_idx = np.vectorize(lambda x_: q_idx.index(x_) + num_support, otypes=[np.long])
        # query_idx = torch.from_numpy(transfrom_idx(query_idx_)).to(self.args.device)

        label = torch.arange(self.args.way, dtype=torch.long).repeat(
            self.args.num_tasks * self.args.query  # *(self.train_loader.num_device if args.multi_gpu else 1)
        ).to(self.args.device)
        logit1, _ = self.model._forward(instance_embs, support_idx_, query_idx_)
        logit2, _ = self.model._forward(instance_embs, support_idx_, query_idx_[perm])
        loss1 = F.cross_entropy(logit1, label)
        loss2 = F.cross_entropy(logit2, label)
        loss = c * loss1 + (1 - c) * loss2
        # print(loss)
        return logits, loss

    def _forward(self, x, support_idx, query_idx):
        pass
