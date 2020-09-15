import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel


# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class MeanNet(FewShotModel):

    def _forward_task(self, emb_dim, support, query):
        num_batch = support.shape[0]
        num_way = support.shape[2]
        num_support = np.prod(support.shape[1:3])
        num_query = np.prod(query.shape[1:3])
        # support: (num_batch, num_shot * num_way, num_emb)
        support = support.view(num_batch, -1, emb_dim)  # Ntask x NK x d
        # query: (num_batch, num_query* num_way, num_emb)
        query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

        if self.args.use_euclidean:
            support = support.unsqueeze(1)
            query = query.unsqueeze(2)
            logits = - torch.sum((support - query) ** 2, dim=-1)
        else:
            support = F.normalize(support, dim=-1)  # normalize for cosine distance

            logits = torch.bmm(query, support.permute([0, 2, 1]))
        logits = torch.mean(logits.view(num_batch, num_query, -1, num_way), dim=2) / self.args.temperature
        logits = logits.view(-1, num_way)

        if self.training:
            return logits, None
        else:
            return logits
