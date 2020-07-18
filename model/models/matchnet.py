import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel
from model.utils import one_hot


# Note: This is the MatchingNet without FCE
#       it predicts an instance based on nearest neighbor rule (not Nearest center mean)

class MatchNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward_task(self, emb_dim, support, query):
        if self.training:
            label_support = torch.arange(self.args.way).repeat(self.args.shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.way)
        else:
            label_support = torch.arange(self.args.eval_way).repeat(self.args.eval_shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.eval_way)
        if torch.cuda.is_available():
            label_support_onehot = label_support_onehot.cuda()  # KN x N

        num_batch = support.shape[0]
        num_way = support.shape[2]
        num_support = np.prod(support.shape[1:3])
        num_query = np.prod(query_idx.shape[-2:])
        support = support.view(num_batch, num_support, emb_dim)  # Ntask x NK x d
        label_support_onehot = label_support_onehot.unsqueeze(0).repeat(num_batch, 1, 1)
        query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean:
            support = support.unsqueeze(1)
            query = query.unsqueeze(2)
            logits = - torch.sum((support - query) ** 2, dim=-1)
        else:
            support = F.normalize(support, dim=-1)  # normalize for cosine distance

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, support.permute([0, 2, 1]))
        logits = torch.bmm(logits, label_support_onehot) / self.args.temperature  # KqN x N
        logits = logits.view(-1, num_way)

        if self.training:
            return logits, None
        else:
            return logits
