import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from model.models.protonet import ProtoNet
import copy
from model.utils import get_summary_writer


# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

class MemoryBankProto(ProtoNet):
    def __init__(self, args):
        super().__init__(args)
        self.stride = args.batch_size
        self.encoder_q = self.encoder
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.writer = get_summary_writer()

        for p in self.encoder_k.parameters():
            p.requires_grad = False
        self.m = args.m
        self.K = args.K
        self.Q = int(args.way * args.bank_ratio)
        self.register_buffer("support_queue", torch.randn(self.K * self.stride, self.hdim))
        if self.Q > 0:
            self.register_buffer("query_queue", torch.randn(self.K * self.stride, *args.image_shape))
            self.register_buffer("idx_queue", torch.zeros(self.K * self.stride).long())
        # self.queue = torch.randn(self.K, self.hdim).requires_grad_(False)
        self.queue_ptr = 0
        self.choice_idx = np.arange(args.batch_size)
        self.count = 0

    # @profile
    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs_q = self.encoder_q(x)
            with torch.no_grad():
                self._momentum_update_key_encoder()
                instance_embs_k = self.encoder_k(x)

            # num_inst = instance_embs.shape[0]
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                logits, logits_reg = self._forward_emb(instance_embs_q, instance_embs_k, support_idx, query_idx)
                repeat = self.args.shot + self.args.query
                shape = (repeat, self.args.batch_size) if self.args.unsupervised else (repeat, self.args.num_classes)
                keys = instance_embs_k.view(*shape, -1)
                images = x.view(*shape, *x.shape[1:])
                s = keys[0, :]
                # idx = np.random.choice(self.choice_idx
                q = images[1, :]
                self._dequeue_and_enqueue(s, q, self.choice_idx)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs_q, support_idx, query_idx)
                return logits

    # self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    # @profile
    def _forward_emb(self, instance_embs_q, instance_embs_k, support_idx, query_idx):
        if not self.training or self.count <= 0:
            return super()._forward(instance_embs_q, support_idx, query_idx)

        emb_dim = instance_embs_q.size(-1)

        # organize support/query data
        support = instance_embs_k[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
        query = instance_embs_q[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

        # get mean of the support
        raw_proto = support.mean(dim=1)  # Ntask x NK x d
        d_s = self.support_queue[:self.count * self.stride]
        dummy = d_s.expand(raw_proto.size(0), *d_s.shape)
        proto = torch.cat([raw_proto, dummy], dim=1)

        num_query = np.prod(query_idx.shape[-2:])
        # query: (num_batch, num_query, num_proto, num_emb)
        # proto: (num_batch, num_proto, num_emb)
        if self.args.use_euclidean:
            logits = self.euclidean(emb_dim, num_query, proto, query)
        else:  # cosine similarity: more memory efficient
            logits = self.cosine(emb_dim, proto, query)

        # --------------- for dummy query --------------
        if self.Q > 0:
            dummy_choose = np.random.choice(self.count * self.stride, self.args.batch_size, replace=False)
            dummy_query = self.query_queue[dummy_choose]
            dummy_query_emb = self.encoder(dummy_query)
            dummy_index = self.idx_queue[dummy_choose]
            q_idx = np.stack([np.random.choice(self.args.batch_size, self.Q) for _ in range(self.args.num_tasks)])
            d_q = dummy_query_emb[q_idx]
            d_i = dummy_index[q_idx] + self.args.way
            if self.args.use_euclidean:
                dummy_logit = self.euclidean(emb_dim, self.Q, proto, d_q)
            else:
                dummy_logit = self.cosine(emb_dim, proto, d_q)
            dummy_label = d_i.view(-1)
            dummy_loss = F.cross_entropy(dummy_logit, dummy_label)
        else:
            dummy_loss = None
            # -----------------------------------------------
        return logits, dummy_loss


    @torch.no_grad()
    # @profile
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        diff = 0
        for (name_q, param_q), (name_k, param_k) in zip(self.encoder_q.named_parameters(),
                                                        self.encoder_k.named_parameters()):
            # self.writer.add_histogram('encoder_q/%s' % name_q, param_q.clone().cpu().data.numpy(), self.gep)
            # self.writer.add_histogram('encoder_k/%s' % name_k, param_k.clone().cpu().data.numpy(), self.gep)
            diff += torch.sum(torch.abs(param_k.data - param_q.data))
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        diff = diff.item()
        self.writer.add_scalar('difference', diff, self.ep)
        self.writer.add_scalar('difference/epoch%s' % self.gep, diff, self.lep)
        self.ep += 1
        self.lep += 1

    @torch.no_grad()
    def _dequeue_and_enqueue(self, s, q, i):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = s.shape[0]

        # ptr = int(self.queue_ptr)
        ptr = self.queue_ptr
        assert self.args.batch_size == batch_size  # for simplicity

        true_idx = ptr * batch_size + i
        # replace the keys at ptr (dequeue and enqueue)
        self.support_queue[ptr * batch_size:(ptr + 1) * batch_size, :].copy_(s)
        if self.Q > 0:
            self.query_queue[ptr * batch_size:(ptr + 1) * batch_size].copy_(q)
            self.idx_queue[ptr * batch_size:(ptr + 1) * batch_size].copy_(torch.from_numpy(true_idx))
        ptr = (ptr + 1) % self.K  # move pointer

        self.queue_ptr = ptr
        if self.count < self.K:
            self.count += 1

    def load_state_dict(self, state_dict, strict=True):
        self.encoder.load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return self.encoder.state_dict(destination, prefix, keep_vars)