#!/usr/bin/env python

"""
    nn_modules.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from scipy import sparse
from helpers import to_numpy


# --
# Samplers

class UniformNeighborSampler(object):
    """
        Samples from a "dense 2D edgelist", which looks like
        
            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]
        
        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):
        cuda = adj.is_cuda

        neigh = []
        mask = []
        for v in ids:
            nonz = torch.nonzero(adj[v]).view(-1)
            if (len(nonz) == 0):
                # no neighbor, only sample from itself
                # for edge embedding... PADDING with all-zero embedding at edge_emb[0]
                if cuda:
                    neigh.append(torch.cuda.LongTensor([v]).repeat(n_samples))
                else:
                    neigh.append(torch.LongTensor([v]).repeat(n_samples))
            else:
                idx = np.random.choice(nonz.shape[0], n_samples)
                neigh.append(nonz[idx])
        neigh = torch.stack(neigh).long().view(-1)
        edges = adj[
            ids.view(-1, 1).repeat(1, n_samples).view(-1),
            neigh]
        return neigh, edges, mask


class SpUniformNeighborSampler(object):
    """
        Samples from a "sparse 2D edgelist", which looks like

            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]

        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):

        cuda = adj.is_cuda

        nonz = adj._indices()
        values = adj._values()

        mask = []
        neigh = []
        edges = []
        for v in ids:
            n = torch.nonzero(nonz[0, :] == v).view(-1)
            if (len(n) == 0):
                # no neighbor, only sample from itself
                # for edge embedding... PADDING with all-zero embedding at edge_emb[0]
                if cuda:
                    neigh.append(torch.cuda.LongTensor([v]).repeat(n_samples))
                    edges.append(torch.cuda.LongTensor([0]).repeat(n_samples))
                    mask.append(torch.cuda.LongTensor([1]).repeat(n_samples))
                else:
                    neigh.append(torch.LongTensor([v]).repeat(n_samples))
                    edges.append(torch.LongTensor([0]).repeat(n_samples))
                    mask.append(torch.LongTensor([1]).repeat(n_samples))
            else:
                # np.random.choice(nonz.shape[0], n_samples)
                if n.shape[0] >= n_samples:
                    idx = torch.randint(0, n.shape[0], (n_samples,))

                    neigh.append(nonz[1, n[idx]])
                    edges.append(values[n[idx]])
                    if cuda:
                        mask.append(torch.cuda.LongTensor([0]).repeat(n_samples))
                    else:
                        mask.append(torch.LongTensor([0]).repeat(n_samples))

                else:

                    if cuda:
                        neigh.append(torch.cat([nonz[1, n], torch.cuda.LongTensor([v]).repeat(n_samples - n.shape[0])]))
                        edges.append(torch.cat([values[n], torch.cuda.LongTensor([0]).repeat(n_samples - n.shape[0])]))
                        mask.append(torch.cat([torch.cuda.LongTensor([0]).repeat(n.shape[0]),
                                               torch.cuda.LongTensor([1]).repeat(n_samples - n.shape[0])]))
                    else:
                        neigh.append(torch.cat([nonz[1, n], torch.LongTensor([v]).repeat(n_samples - n.shape[0])]))
                        edges.append(torch.cat([values[n], torch.LongTensor([0]).repeat(n_samples - n.shape[0])]))
                        mask.append(torch.cat([torch.LongTensor([0]).repeat(n.shape[0]),
                                               torch.LongTensor([1]).repeat(n_samples - n.shape[0])]))

        neigh = torch.stack(neigh).long().view(-1)
        edges = torch.stack(edges).long().view(-1)
        mask = torch.stack(mask).float()

        return neigh, edges, mask


class DenseMask(object):
    """
        Samples from a "sparse 2D edgelist", which looks like

            [
                [0, 1, 2, ..., 4],
                [0, 0, 5, ..., 10],
                ...
            ]

        stored as torch.LongTensor.

        Adj[a1,a2]=id, where the edge embedding of edge (a1,a2) is stored at emb[id]

        If a node does not have a neighbor, sample itself n_sample times and
        return emb[0]. * emb[0] is the padding zero embedding.

        We didn't apply the optimization from GraphSage
    """

    def __init__(self):
        pass

    def __call__(self, adj, ids, n_samples=16):

        cuda = adj.is_cuda

        neigh = []
        edges = adj[ids]
        if cuda:
            mask = torch.where(adj == 0, torch.cuda.FloatTensor([0]), torch.cuda.FloatTensor([1]))
        else:
            mask = torch.where(adj == 0, torch.FloatTensor([0]), torch.FloatTensor([1]))

        return neigh, edges, mask[ids]


sampler_lookup = {
    "uniform_neighbor_sampler": UniformNeighborSampler,
    "sparse_uniform_neighbor_sampler": SpUniformNeighborSampler,
    "dense_mask": DenseMask,
}


# --
# Preprocessers

class IdentityPrep(nn.Module):
    def __init__(self, input_dim, n_nodes=None, embedding_dim=64):
        """ Example of preprocessor -- doesn't do anything """
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim

    @property
    def output_dim(self):
        return self.input_dim

    def forward(self, ids, feats, layer_idx=0):
        return feats


class NodeEmbeddingPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, pre_trained=None, embedding_dim=64):
        """ adds node embedding """
        super(NodeEmbeddingPrep, self).__init__()

        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # Affine transform, for changing scale + location

        if pre_trained is not None:
            self.embedding.from_pretrained(pre_trained, padding_idx=0)

    @property
    def output_dim(self):
        if self.input_dim:
            return self.input_dim + self.embedding_dim
        else:
            return self.embedding_dim

    def forward(self, ids, feats, layer_idx=0):
        if layer_idx > 0:
            embs = self.embedding(ids)
        else:
            # Don't look at node's own embedding for prediction, or you'll probably overfit a lot
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))

        embs = self.fc(embs)
        if self.input_dim and feats is not None:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs


class LinearPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, output_dim=32, embedding_dim=64):
        """ adds node embedding """
        super(LinearPrep, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim = output_dim

    def forward(self, ids, feats, layer_idx=0):
        return self.fc(feats)


# --
# Aggregators

class AggregatorMixin(object):
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)


class MeanAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))  # !! Careful
        agg_neib = agg_neib.mean(dim=1)  # Careful

        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)

        return out


class PoolAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, pool_fn, activation, hidden_dim=512,
                 combine_fn=lambda x: torch.cat(x, dim=1)):
        super(PoolAggregator, self).__init__()

        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)

        self.output_dim_ = output_dim
        self.activation = activation
        self.pool_fn = pool_fn
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = self.pool_fn(agg_neib)

        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)

        return out


class MaxPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MaxPoolAggregator, self).__init__(**{
            "input_dim": input_dim,
            "output_dim": output_dim,
            "pool_fn": lambda x: x.max(dim=1)[0],
            "activation": activation,
            "hidden_dim": hidden_dim,
            "combine_fn": combine_fn,
        })


class MeanPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanPoolAggregator, self).__init__(**{
            "input_dim": input_dim,
            "output_dim": output_dim,
            "pool_fn": lambda x: x.mean(dim=1),
            "activation": activation,
            "hidden_dim": hidden_dim,
            "combine_fn": combine_fn,
        })


class LSTMAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation,
                 hidden_dim=512, bidirectional=False, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(LSTMAggregator, self).__init__()
        assert not hidden_dim % 2, "LSTMAggregator: hiddem_dim % 2 != 0"

        self.lstm = nn.LSTM(input_dim, hidden_dim // (1 + bidirectional), bidirectional=bidirectional, batch_first=True)
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)

        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn

    def forward(self, x, neibs):
        x_emb = self.fc_x(x)

        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(agg_neib)
        agg_neib = agg_neib[:, -1, :]  # !! Taking final state, but could do something better (eg attention)
        neib_emb = self.fc_neib(agg_neib)

        out = self.combine_fn([x_emb, neib_emb])
        if self.activation:
            out = self.activation(out)

        return out


class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=32,
                 dropout=0.5, alpha=0.8,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(AttentionAggregator, self).__init__()

        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm
        self.output_dim = output_dim
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # Compute attention weights
        neib_att = self.att(neibs)
        x_att = self.att(x)
        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        x_att = x_att.view(x_att.size(0), x_att.size(1), 1)
        # ws = F.softmax(torch.bmm(neib_att, x_att).squeeze())

        ws = torch.bmm(neib_att, x_att).squeeze()
        ws += -9999999 * mask
        ws = F.softmax(ws, dim=1)

        # Weighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib * ws.unsqueeze(-1), dim=1)

        out = self.fc_x(x) + self.fc_neib(agg_neib)

        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        if self.activation:
            out = self.activation(out)

        return out


class EdgeEmbAttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, dropout=0.5, alpha=0.8,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(EdgeEmbAttentionAggregator, self).__init__()
        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm
        self.alpha = alpha
        self.concat_node = concat_node
        if concat_node:
            self.output_dim = 2 * output_dim
        else:
            self.output_dim = output_dim
        if concat_edge:
            self.output_dim += edge_dim
        self.concat_edge = concat_edge

        self.activation = activation
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(W.data, gain=1.414)
        self.register_parameter('W', W)

        W2 = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(W2.data, gain=1.414)
        self.register_parameter('W2', W2)

        a = nn.Parameter(torch.zeros(size=(2 * output_dim + edge_dim, 1)))
        nn.init.xavier_uniform_(a.data, gain=1.414)
        self.register_parameter('a', a)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, input, neigh_feat, edge_emb, mask):
        # Compute attention weights
        N = input.size()[0]

        x = torch.mm(input, self.W)
        neighs = torch.mm(neigh_feat, self.W2)

        n_sample = int(neighs.shape[0] / x.shape[0])

        a_input = torch.cat([x.repeat(1, n_sample).view(N, n_sample, -1),
                             neighs.view(N, n_sample, -1),
                             edge_emb.view(N, n_sample, -1)], dim=2)

        e = self.leakyrelu(torch.matmul(a_input, self.a))
        e += -9999999 * mask
        attention = F.softmax(e, dim=1)
        attention = attention.view(N, 1, n_sample)
        # attention = attention.squeeze(2)

        # h_prime = [torch.matmul(attention[i], neigh_feat.view(N, n_sample, -1)[i]) for i in range(N)]
        h_prime = torch.bmm(attention, neighs.view(N, n_sample, -1)).squeeze()

        h_prime = self.dropout(h_prime)

        if self.batchnorm:
            h_prime = self.bn(h_prime)

        if self.concat_node:
            output = torch.cat([x, h_prime], dim=1)
        else:
            output = h_prime + x

        if self.concat_edge:
            e = torch.bmm(attention, edge_emb.view(N, n_sample, -1)).squeeze()
            output = torch.cat([output, e], dim=1)
        if self.activation:
            output = self.activation(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' + ' + str(self.edge_dim) \
               + ' -> ' + str(self.output_dim) + ')'


class AttentionAggregator2(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=32,
                 dropout=0.5,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(AttentionAggregator2, self).__init__()

        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.att2 = nn.Sequential(*[
            nn.Linear(input_dim + edge_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim + edge_dim, output_dim, bias=False)
        self.concat_node = concat_node

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        if concat_node:
            self.output_dim = output_dim * 2
        else:
            self.output_dim = output_dim
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # Compute attention weights
        neibs = torch.cat([neibs, edge_emb], dim=1)

        neib_att = self.att2(neibs)
        x_att = self.att(x)

        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        x_att = x_att.view(x_att.size(0), x_att.size(1), 1)

        ws = torch.bmm(neib_att, x_att).squeeze()
        ws += -9999999 * mask
        ws = F.softmax(ws, dim=1)

        # Weighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib * ws.unsqueeze(-1), dim=1)

        if self.concat_node:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)], dim=1)
        else:
            out = self.fc_x(x) + self.fc_neib(agg_neib)

        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        if self.activation:
            out = self.activation(out)

        return out


class AttentionAggregator3(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, activation, hidden_dim=32,
                 dropout=0.5,
                 concat_node=True, concat_edge=True, batchnorm=False):
        super(AttentionAggregator3, self).__init__()

        self.att_x = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(hidden_dim,1)
        ])

        self.att_neigh = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, 1)
        ])

        self.att_edge = nn.Sequential(*[
            nn.Linear(edge_dim, hidden_dim, bias=True),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            # nn.ReLU(),
            # nn.Linear(hidden_dim,1)
        ])

        self.fc_x = nn.Linear(2 * input_dim, output_dim)
        self.fc_neib = nn.Linear(input_dim, output_dim)
        self.fc_edge = nn.Linear(edge_dim, output_dim)
        self.concat_node = concat_node

        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        if concat_node:
            self.output_dim = output_dim * 2
        else:
            self.output_dim = output_dim
        self.activation = activation

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.output_dim)

    def forward(self, x, neibs, edge_emb, mask):
        '''
        :param x: (N,input_dim)
        :param neibs: (n_nodes,input_dim)
        :param edge_emb: (N*n_nodes,edge_dim)
        :param mask: (N, n_nodes)
        :return:
        '''
        N = x.shape[0]

        neib_att = self.att_neigh(neibs)
        x_att = self.att_x(x)
        edge_att = self.att_edge(edge_emb)

        ws = x_att.mm(neib_att.t())  # +edge_att.view(N,-1)
        # ws = F.leaky_relu(ws)
        ws = ws * mask
        ws = F.softmax(ws, dim=1)

        # Weighted average of neighbors
        agg_neib = torch.mm(ws, neibs)
        agg_neib = F.sigmoid(agg_neib)
        # agg_edge = edge_emb.view(N, -1, edge_emb.size(-1))
        # agg_edge = torch.sum(agg_edge * ws.unsqueeze(-1), dim=1)

        if self.concat_node:
            out = torch.cat([self.fc_x(x), self.fc_neib(agg_neib)],dim=1)
        else:
            # out = self.fc_x(x) + self.fc_neib(agg_neib)

            out = torch.cat([x, agg_neib], dim=1)
            out = self.fc_x(out)
            out = F.sigmoid(out)
            
        if self.batchnorm:
            out = self.bn(out)

        out = self.dropout(out)

        # if self.activation:
        #     out = self.activation(out)

        return out


class EdgeAggregator(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(EdgeAggregator, self).__init__()

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        W1 = nn.Parameter(torch.zeros(size=(input_dim, edge_dim)))
        nn.init.xavier_uniform_(W1.data, gain=1.414)
        self.register_parameter('W1', W1)

        W2 = nn.Parameter(torch.zeros(size=(edge_dim, edge_dim)))
        nn.init.xavier_uniform_(W2.data, gain=1.414)
        self.register_parameter('W2', W2)

        B = nn.Parameter(torch.zeros(size=(1, edge_dim)))
        nn.init.xavier_uniform_(B.data, gain=1.414)
        self.register_parameter('B', B)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.edge_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # update edge embedding:
        # e = sigma(w1*x+W2*neibs+b) @ e

        # self.W1.to(x.device)
        # self.W2.to(x.device)
        # self.B.to(x.device)

        n = edge_emb.shape[0]
        n_sample = int(edge_emb.shape[0] / x.shape[0])

        x_input = torch.mm(x.repeat(n_sample, 1), self.W1)

        n_input = torch.mm(neibs, self.W1)

        e_input = torch.mm(edge_emb, self.W2)

        a_input = e_input + n_input + x_input + self.B.repeat(n, 1)

        a_input = self.dropout(a_input)

        if self.batchnorm:
            a_input = self.bn(a_input)

        if self.activation:
            a_input = self.activation(a_input)

        emb = a_input * edge_emb

        return emb

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' + ' + str(self.edge_dim) \
               + ' -> ' + str(self.edge_dim) + ')'


class IdEdgeAggregator(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(IdEdgeAggregator, self).__init__()

        self.input_dim = input_dim
        self.activation = activation
        self.edge_dim = edge_dim
        self.batchnorm = batchnorm
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, neibs, edge_emb, mask):
        # identical mapping
        # e = sigma(w1*x+W2*neibs+b) @ e
        return edge_emb


class ResEdge(nn.Module):
    def __init__(self, input_dim, edge_dim, activation, dropout=0.5, batchnorm=False):
        super(ResEdge, self).__init__()

        self.input_dim = input_dim
        self.edge_dim = edge_dim
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        W1 = nn.Parameter(torch.zeros(size=(input_dim, edge_dim)))
        nn.init.xavier_uniform_(W1.data, gain=1.414)
        self.register_parameter('W1', W1)

        W2 = nn.Parameter(torch.zeros(size=(edge_dim, edge_dim)))
        nn.init.xavier_uniform_(W2.data, gain=1.414)
        self.register_parameter('W2', W2)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.edge_dim)

    def forward(self, x, neibs, edge_emb, mask):
        # update edge embedding:
        # e = sigma(W1*x+W1*neibs+W2*e) + e

        # n = edge_emb.shape[0]
        # self.W1.to(x.device)
        # self.W2.to(x.device)

        n_sample = int(edge_emb.shape[0] / x.shape[0])

        x_input = torch.mm(x, self.W1).repeat(n_sample, 1)

        n_input = torch.mm(neibs, self.W1)

        e_input = torch.mm(edge_emb, self.W2)

        a_input = e_input + n_input + x_input

        a_input = self.dropout(a_input)

        if self.batchnorm:
            a_input = self.bn(a_input)

        if self.activation:
            a_input = self.activation(a_input)

        emb = a_input + edge_emb

        return emb

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' + ' + str(self.edge_dim) \
               + ' -> ' + str(self.edge_dim) + ')'


class MetapathAggrLayer(nn.Module):
    """
    metapath attention layer.
    """

    def __init__(self, in_features, alpha=0.8, dropout=0.5, hidden_dim=64, batchnorm=False):
        super(MetapathAggrLayer, self).__init__()
        # self.dropout = dropout
        self.input_dim = in_features
        self.output_dim = in_features
        self.alpha = alpha
        self.dropout = nn.Dropout(p=dropout)
        self.batchnorm = batchnorm

        self.att = nn.Sequential(*[
            nn.Linear(in_features, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])

        self.mlp = nn.Sequential(*[
            nn.Linear(hidden_dim, in_features),
            nn.ReLU(),
            nn.Linear(in_features, in_features),
        ])

        a = nn.Parameter(torch.zeros(size=(hidden_dim, 1)))
        nn.init.xavier_uniform_(a.data, gain=1.414)
        self.register_parameter('a', a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if self.batchnorm:
            self.bn = nn.BatchNorm1d(self.out_features)

    def forward(self, input):
        # input: tensor(nmeta,N,in_features)
        # self.a.to(input.device)

        n_meta = input.shape[0]
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]
        input_dim = input.shape[2]
        input = input.contiguous()
        input = self.att(input.view(-1, input_dim)) \
            .view(N, n_meta, -1)

        # a_input = torch.cat([input.repeat(1,1,self.nmeta).view(N, self.nmeta*self.nmeta, -1),
        #                      input.repeat(1,self.nmeta, 1)], dim=2).view(N, -1, 2 * self.in_features)
        e = self.leakyrelu(torch.matmul(input, self.a).squeeze(2))
        e = F.softmax(e, dim=1).view(N, 1, n_meta)

        output = torch.bmm(e, input).squeeze()

        output = self.dropout(output)

        if self.batchnorm:
            output = self.bn(output)

        weight = torch.sum(e.view(N, n_meta), dim=0) / N

        return F.relu(self.mlp(output)), weight.cpu().detach().numpy()

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'


prep_lookup = {
    "identity": IdentityPrep,
    "node_embedding": NodeEmbeddingPrep,
    "linear": LinearPrep,
}

aggregator_lookup = {
    "mean": MeanAggregator,
    "max_pool": MaxPoolAggregator,
    "mean_pool": MeanPoolAggregator,
    "lstm": LSTMAggregator,
    "attention": AttentionAggregator,
    "attention2": AttentionAggregator2,
    "attention3": AttentionAggregator3,
    "edge_emb_attn": EdgeEmbAttentionAggregator,
}

metapath_aggregator_lookup = {
    "attention": MetapathAggrLayer,
}

edge_aggregator_lookup = {
    "identity": IdEdgeAggregator,
    "attention": EdgeAggregator,
    "residual": ResEdge,
}

import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj = adj
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.mm(input, self.weight)
        output = torch.spmm(self.adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
