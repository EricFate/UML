import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models.protonet import ProtoNet


# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class ExtremeProto(ProtoNet):
    def __init__(self, args):
        super().__init__(args)
        self.K = 640
        self.register_buffer("queue", torch.randn(self.K, self.hdim))
        # self.queue = torch.randn(self.K, self.hdim).requires_grad_(False)
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.queue_ptr = 0
        # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def _forward(self, instance_embs, support_idx, query_idx):
        if not self.training:
            return super()._forward(instance_embs, support_idx, query_idx)

        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

        # get mean of the support
        raw_proto = support.mean(dim=1)  # Ntask x NK x d
        dummy = self.queue.expand(raw_proto.size(0), *self.queue.shape)
        proto = torch.cat([raw_proto, dummy], dim=1)
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean:  # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
        else:  # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
        repeat = self.args.shot + self.args.query
        shape = (repeat, self.args.batch_size) if self.args.unsupervised else (repeat, self.args.num_classes)
        keys = instance_embs.view(*shape, -1)
        keys = keys[0, :]
        self._dequeue_and_enqueue(keys)
        return logits, None

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        # ptr = int(self.queue_ptr)
        ptr = self.queue_ptr
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :].copy_(keys)
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr = ptr
