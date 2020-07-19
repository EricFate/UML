import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel


# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class ProtoNet(FewShotModel):
    def __init__(self, args):
        super().__init__(args)

    def _forward_task(self, emb_dim, support, query):
        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d
        # query: (num_batch, num_query, num_way, num_emb)
        # proto: (num_batch, num_way, num_emb)
        if self.args.use_euclidean:
            logits = self.euclidean(emb_dim, proto, query)
        else:  # cosine similarity: more memory efficient
            logits = self.cosine(emb_dim, proto, query)

        if self.training:
            return logits, None
        else:
            return logits

    def cosine(self, emb_dim, proto, query, max_pool=False):
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
        query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)
        # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
        # logits = torch.einsum('ijk,ilmk->ilmj', proto, query)
        logits = logits.reshape(-1, num_proto)
        if max_pool:
            logits = torch.max(logits, dim=-1, keepdim=True)[0]
        return logits

    def euclidean(self, emb_dim, proto, query, max_pool=False):
        num_query = np.prod(query.shape[1:3])
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
        proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
        proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)
        logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        if max_pool:
            logits = torch.max(logits, dim=-1, keepdim=True)[0]
        return logits
