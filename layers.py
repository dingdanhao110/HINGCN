import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout=0.5, alpha=0.8, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.att_x = nn.Linear(out_features, 1)
        self.att_nei = nn.Linear(out_features, 1)
        self.fc = nn.Linear(in_features, out_features)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, batch=64):
        x = self.fc(input)
        neib_att = self.att_nei(x).t()
        result = []
        adjs = torch.split(adj, batch, dim=0)
        for chunk_id, chunk in enumerate(torch.split(x, batch, dim=0)):
            N = chunk.shape[0]
            x_att = self.att_x(chunk)

            ws = x_att+neib_att

            zero_vec = -9e15*torch.ones_like(ws)
            ws = torch.where(adjs[chunk_id] > 0, ws, zero_vec)
            ws = F.softmax(ws, dim=1)
            # ws = F.dropout(ws, 0.6, training=self.training)

            # Weighted average of neighbors
            agg_neib = torch.mm(ws, x)
            #agg_neib = F.sigmoid(agg_neib)
            # agg_edge = edge_emb.view(N, -1, edge_emb.size(-1))
            # agg_edge = torch.sum(agg_edge * ws.unsqueeze(-1), dim=1)

            out = chunk + agg_neib
            out = F.elu(out)
            result.append(out)
        result = torch.cat(result, dim=0)
        result = self.dropout(result)
        return result

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphInductiveLayer(Module):
    """
    Graph inductive attention layer, similar function to GCN. Attention+GraphSage
    """

    def __init__(self, in_features, out_features,
                 nsamples,
                 sampler_class, adj, train_adj,
                 aggr_class,
                 bias=True):
        super(GraphInductiveLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.sampler = sampler_class(adj=adj)
        self.train_sampler = sampler_class(adj=train_adj)

        self.aggregator = aggr_class(input_dim=in_features,
                                     output_dim=out_features, nsamples=nsamples)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, train=True):
        ids = input[:, 0]
        # Sample neighbors
        sample_fns = self.train_sampler if train else self.sampler

        neighbor_adj = sample_fns(ids)

        input = self.aggregator(input, neighbor_adj)

        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


# class MetapathAttentionLayer(nn.Module):
#     """
#     metapath attention layer.
#     """
#
#     def __init__(self, in_features, nmeta, dropout, alpha, concat=True):
#         super(MetapathAttentionLayer, self).__init__()
#         self.dropout = dropout
#         self.in_features = in_features
#         self.alpha = alpha
#         self.concat = concat
#
#         # Weight: [in_features][num_meta]
#         self.W = nn.Parameter(torch.zeros(size=(in_features, nmeta)))
#         nn.init.xavier_uniform_(self.W.data, gain=1.414)
#
#     def forward(self, input):
#         """from input[metapath_id][v_id][embedding] to [v_id][embedding]"""
#         attention = torch.FloatTensor(input.size()[1], input.size()[0])  # shape(|vertices|,num_metapath)
#         for mp_id in range(input.size()[0]):  # mp_id: metapath_idx
#             attention[:, mp_id] = torch.mm(input[mp_id, :, :], self.W[:, mp_id])
#         # attention = F.dropout(attention, self.dropout, training=self.training)
#         attention = F.relu(attention)
#         attention = F.softmax(attention, dim=1)
#         # attention = F.dropout(attention, self.dropout, training=self.training)
#
#         output = torch.FloatTensor(input.size()[1], input.size()[2])  # shape(|vertices|,embed_len)
#         for v_id in range(input.size()[1]):
#             for mp_id in range(input.size()[0]):
#                 output[v_id, :] = output[v_id, :] + input[mp_id, v_id, :] * attention[v_id, mp_id]
#
#         if self.concat:
#             return F.elu(output)
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'
