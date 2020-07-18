import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models import FewShotModel


class DummyProto(FewShotModel):

    def __init__(self, args):
        super().__init__(args)
        self.dummy = nn.Linear(self.hdim, args.dummy_nodes, bias=False)
        self.activate_counter = torch.zeros(args.dummy_nodes).to(args.device)

    def _forward_dummy(self, instance_embs, dummy_emb, support_idx, query_idx, dummy_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        logits = self._compute_logits(emb_dim, num_batch, num_proto, num_query, proto, query)
        dummy = dummy_emb[dummy_idx.flatten()].view(*(dummy_idx.shape + (-1,)))
        dummy_logits = self._compute_logits(emb_dim, num_batch, num_proto, num_query, proto, dummy)
        dummy_label = torch.zeros((dummy_logits.size(0))).long().to(self.args.device)
        dummy_label.fill_(self.args.way)
        dummy_loss = F.cross_entropy(dummy_logits, dummy_label)
        return logits, dummy_loss

    def _forward(self, instance_embs, support_idx, query_idx):
        emb_dim = instance_embs.size(-1)

        # organize support/query data
        support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d
        num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        num_query = np.prod(query_idx.shape[-2:])
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        logits = self._compute_logits(emb_dim, num_batch, num_proto, num_query, proto, query)
        return logits

    def _compute_logits(self, emb_dim, num_batch, num_proto, num_query, proto, query):
        if self.args.use_euclidean:  # self.args.use_euclidean:
            query = query.view(-1, emb_dim).unsqueeze(1)  # (Nbatch*Nq*Nw, 1, d)
            proto = proto.unsqueeze(1).expand(num_batch, num_query, num_proto, emb_dim)
            proto = proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)  # (Nbatch x Nq, Nk, d)

            logits = - torch.sum((proto - query) ** 2, 2) / self.args.temperature
            # dummy logits
            dummy_shape = self.dummy.weight.shape
            dummy_proto = self.dummy.weight \
                .view((1, 1,) + dummy_shape) \
                .expand(num_batch, num_query, num_proto, emb_dim)
            dummy_proto = dummy_proto.contiguous().view(num_batch * num_query, num_proto, emb_dim)
            dummy_logits = - torch.sum((dummy_proto - query) ** 2, 2) / self.args.temperature
        else:  # cosine similarity: more memory efficient
            proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
            query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)

            # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
            logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
            logits = logits.view(-1, num_proto)
            # dummy logits
            dummy_proto = F.normalize(self.dummy.weight, dim=-1)
            num_dummy = dummy_proto.size(0)
            dummy_proto = dummy_proto.unsqueeze(0).expand(num_batch, num_dummy, emb_dim)
            dummy_logits = torch.bmm(query, dummy_proto.permute([0, 2, 1])) / self.args.temperature
            dummy_logits = dummy_logits.view(-1, num_dummy)

        tmp = torch.max(dummy_logits, dim=-1, keepdim=True)
        dummy_logit = tmp[0]
        self.activate_counter[tmp[1]] += 1
        logits = torch.cat([logits, dummy_logit], dim=1)
        return logits

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            if self.training:
                x, dummy = x
            instance_embs = self.encoder(x)
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                dummy_emb = self.encoder(dummy)
                # sample dummy
                dummy_size = dummy_emb.size(0)
                dummy_range = np.arange(dummy_size)
                dummy_idx = torch.from_numpy(
                    np.stack([np.random.choice(dummy_range, self.args.query, replace=False) for _ in
                              range(self.args.num_tasks)])) \
                    .unsqueeze(2).long().to(self.args.device)
                logits, dummy_loss = self._forward_dummy(instance_embs, dummy_emb, support_idx, query_idx, dummy_idx)
                return logits, dummy_loss
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits
