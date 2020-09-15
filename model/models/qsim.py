from abc import ABC

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.models import FewShotModel
from model.utils import one_hot, get_dataset
from PIL import Image
from tqdm import tqdm


# Note: As in Protonet, we use Euclidean Distances here, you can change to the Cosine Similarity by replace
#       TRUE in line 30 as self.args.use_euclidean

def minus(pq_sim, pn_sim, qn_sim):
    return pq_sim - pn_sim - qn_sim


def multiply(pq_sim, pn_sim, qn_sim):
    return pq_sim * (pn_sim + qn_sim) / 2


def divide(pq_sim, pn_sim, qn_sim):
    return pq_sim / ((pn_sim + qn_sim) / 2)


class QsimBase(FewShotModel, ABC):

    def __init__(self, args):
        super().__init__(args)

        trainset = get_dataset(self.args.dataset, 'train', False, self.args, augment='none')
        self.loader = DataLoader(dataset=trainset, batch_size=256, shuffle=False,
                                 num_workers=self.args.num_workers,
                                 pin_memory=True)
        self.cosine = nn.CosineSimilarity(dim=-1)

    def loading_negatives(self):

        embs = []
        with torch.no_grad():
            for i, batch in enumerate(self.loader, 1):
                if torch.cuda.is_available():
                    data = batch[0].cuda()
                else:
                    data, _ = batch
                data_emb = self.encoder(data)
                embs.append(data_emb)
        self.neg_embs = torch.cat(embs)

    def hard_mining(self, query: torch.Tensor):
        """
        :param query: (num_tasks,num_query,num_way,num_emb)
        :return:
        """
        query = query.unsqueeze(3)
        factor = np.prod(query.shape[:3]) / 75
        if factor > 1:
            sampled_number = int(self.neg_embs.size(0) / factor)
            sampled_idx = np.random.choice(self.neg_embs.size(0), sampled_number)
            embs = self.neg_embs[sampled_idx]
        else:
            embs = self.neg_embs
        neg_embs = embs.view(1, 1, 1, *embs.shape)
        if self.args.use_euclidean:
            sims = - torch.sum((query - neg_embs) ** 2, dim=-1)
        else:  # cosine similarity: more memory efficient
            sims = self.cosine(query, neg_embs)
        _, negative_idx = torch.topk(sims, k=self.args.num_negative, dim=-1, largest=False)
        negatives = self.neg_embs[negative_idx]
        return negatives

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            x = x.squeeze(0)
            instance_embs = self.encoder(x)
            # split support query set for few-shot data

            support_idx, query_idx = self.split_instances(instance_embs)

            support = instance_embs[support_idx.flatten()].view(*(support_idx.shape + (-1,)))
            query = instance_embs[query_idx.flatten()].view(*(query_idx.shape + (-1,)))

            if not hasattr(self, 'neg_embs'):
                self.loading_negatives()

            if self.args.hard_mining:
                negatives = self.hard_mining(query)
            else:
                indexes = torch.from_numpy(
                    np.stack([np.random.choice(len(self.neg_embs), self.args.num_negative, replace=False) for i in
                              range(np.prod(query.shape[:3]))]).reshape(
                        (*query.shape[:3], self.args.num_negative)))\
                    .long().to(self.args.device)
                negatives = self.neg_embs[indexes]

            logits = self._forward_qsim(support, query, negatives)
            return logits

    def _forward_qsim(self, support, query, negatives):
        raise NotImplementedError


class QsimProtoNet(QsimBase):

    def _forward_qsim(self, support, query, negatives):

        # organize support/query data
        # (num_task,num_way,emb_dim)
        # get mean of the support
        proto = support.mean(dim=1)  # Ntask x NK x d
        # query: (num_batch, num_query, num_way, num_emb)
        # proto: (num_batch, num_way, num_emb)

        if self.args.use_euclidean:
            logits = self.qsim_euclidean(proto, query, negatives)
        else:  # cosine similarity: more memory efficient
            logits = self.qsim_cosine(proto, query, negatives)

        if self.training:
            return logits, None
        else:
            return logits

    def qsim_cosine(self, proto, query, negatives, max_pool=False):
        """
        :param proto: (num_task,num_way,emb_dim)
        :param query: (num_task,num_query,num_way,emb_dim)
        :param negatives: (num_task,num_query,num_way,num_neg,emb_dim)
        :param max_pool:
        :return:
        """
        # num_batch = proto.shape[0]
        num_proto = proto.shape[1]
        proto = F.normalize(proto, dim=-1)  # normalize for cosine distance
        query = F.normalize(query, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        # query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)
        # (num_batch,  num_emb, num_proto) * (num_batch, num_query*num_proto, num_emb) -> (num_batch, num_query*num_proto, num_proto)
        # logits = torch.bmm(query, proto.permute([0, 2, 1])) / self.args.temperature
        # logits = torch.einsum('ijk,ilmk->ilmj', proto, query)
        pq_sim = torch.einsum('ijk,ilmk->ilmj', proto, query)
        pn_sim = torch.einsum('ijk,ilmpk->ilmjp', proto, negatives)
        qn_sim = torch.einsum('ilmk,ilmpk->ilmp', query, negatives)
        # torch.unsqueeze()
        pq_sim = pq_sim.view(*pq_sim.shape, 1)
        pn_sim = pn_sim.view(pn_sim.shape[0], 1, 1, *pn_sim.shape[1:])
        qn_sim = qn_sim.view(*qn_sim.shape[:-1], 1, qn_sim.shape[-1])
        # logits = logits.reshape(-1, num_proto)
        # (num_task,num_query,num_way,num_way,num_neg)
        logits = eval(self.args.qsim_method)(pq_sim, pn_sim, qn_sim)
        logits = torch.mean(logits, dim=-1).view(-1, num_proto)
        if max_pool:
            logits = torch.max(logits, dim=-1, keepdim=True)[0]
        return logits

    def qsim_euclidean(self, proto, query, negatives, max_pool=False):
        """
        :param proto: (num_task,num_way,emb_dim)
        :param query: (num_task,num_query,num_way,emb_dim)
        :param negatives: (num_task,num_query,num_way,num_neg,emb_dim)
        :param max_pool:
        :return:
        """
        num_proto = proto.shape[1]
        proto = proto.view(proto.shape[0], 1, 1, proto.shape[1], 1, proto.shape[2])
        query = query.view(*query.shape[:3], 1, 1, query.shape[-1])
        negatives = negatives.view(*negatives.shape[:3], 1, *negatives.shape[3:])
        pq_sim = 1. / torch.norm(proto - query, dim=-1)
        pn_sim = 1. / torch.norm(proto - negatives, dim=-1)
        qn_sim = 1. / torch.norm(query - negatives, dim=-1)
        logits = eval(self.args.qsim_method)(pq_sim, pn_sim, qn_sim)
        logits = torch.sum(logits, dim=-1).view(-1, num_proto)
        if max_pool:
            logits = torch.max(logits, dim=-1, keepdim=True)[0]
        return logits


class QsimMatchNet(QsimBase):

    def _forward_qsim(self, support, query, negatives):

        num_way = support.shape[2]
        if self.training:
            label_support = torch.arange(self.args.way).repeat(self.args.shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.way)
        else:
            label_support = torch.arange(self.args.eval_way).repeat(self.args.eval_shot).type(torch.LongTensor)
            label_support_onehot = one_hot(label_support, self.args.eval_way)
        label_support_onehot = label_support_onehot.to(self.args.device)  # KN x N
        # get mean of the support
        # query: (num_batch, num_query, num_way, num_emb)
        # proto: (num_batch, num_way, num_emb)
        if self.args.use_euclidean:
            logits = self.qsim_euclidean_match(support, query, negatives)
        else:  # cosine similarity: more memory efficient
            logits = self.qsim_cosine_match(support, query, negatives)

        num_query = np.prod(logits.shape[1:3])
        num_support = np.prod(logits.shape[3:])
        # (num_task,num_query,num_way,num_support,num_way)
        logits = logits.view(logits.shape[0], num_query, num_support)
        logits = torch.einsum('ijk,kl->ijl', logits, label_support_onehot) / self.args.temperature
        logits = logits.view(-1, num_way)

        if self.training:
            return logits, None
        else:
            return logits

    def qsim_cosine_match(self, support, query, negatives):
        """
        :param support: (num_task,num_support,num_way,emb_dim)
        :param query: (num_task,num_query,num_way,emb_dim)
        :param negatives: (num_task,num_query,num_way,num_neg,emb_dim)
        :param max_pool:
        :return:
        """
        # num_batch = proto.shape[0]

        support = F.normalize(support, dim=-1)  # normalize for cosine distance
        query = F.normalize(query, dim=-1)
        negatives = F.normalize(negatives, dim=-1)
        # query = query.view(num_batch, -1, emb_dim)  # (Nbatch,  Nq*Nw, d)
        pq_sim = torch.einsum('ijnk,ilmk->ilmjn', support, query)
        pn_sim = torch.einsum('ijnk,ilmpk->ilmjnp', support, negatives)
        qn_sim = torch.einsum('ilmk,ilmpk->ilmp', query, negatives)
        # torch.unsqueeze()
        pq_sim = pq_sim.view(*pq_sim.shape, 1)
        pn_sim = pn_sim.view(pn_sim.shape[0], 1, 1, *pn_sim.shape[1:])
        qn_sim = qn_sim.view(*qn_sim.shape[:-1], 1, 1, qn_sim.shape[-1])
        # logits = logits.reshape(-1, num_proto)
        # (num_task,num_query,num_way,num_support,num_way,num_neg)
        logits = eval(self.args.qsim_method)(pq_sim, pn_sim, qn_sim)
        # (num_task,num_query,num_way,num_support,num_way)
        logits = torch.mean(logits, dim=-1)
        return logits

    def qsim_euclidean_match(self, support, query, negatives):
        """
        :param support: (num_task,num_support,num_way,emb_dim)
        :param query: (num_task,num_query,num_way,emb_dim)
        :param negatives: (num_task,num_neg,emb_dim)
        :param max_pool:
        :return:
        """
        # (num_task,num_query,num_way,num_support,num_way,num_neg)
        support = support.view(support.shape[0], 1, 1, *support.shape[1:-1], 1, support.shape[-1])
        query = query.view(*query.shape[:3], 1, 1, 1, query.shape[-1])
        negatives = negatives.view(*negatives.shape[:3], 1, *negatives.shape[3:])
        pq_sim = 1. / torch.norm(support - query, dim=-1)
        pn_sim = 1. / torch.norm(support - negatives, dim=-1)
        qn_sim = 1. / torch.norm(query - negatives, dim=-1)
        logits = eval(self.args.qsim_method)(pq_sim, pn_sim, qn_sim)
        logits = torch.sum(logits, dim=-1)
        return logits
