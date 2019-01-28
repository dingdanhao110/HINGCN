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

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)  # mask; only neighbors contribute to weight
        attention = torch.where(adj > 0, e, zero_vec)  # mask; only neighbors contribute to weight
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MetapathAttentionLayer(nn.Module):
    """
    metapath attention layer.
    """

    def __init__(self, in_features, nmeta, dropout, alpha, concat=True):
        super(MetapathAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.alpha = alpha
        self.concat = concat

        # Weight: [in_features][num_meta]
        self.W = nn.Parameter(torch.zeros(size=(in_features, nmeta)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

    def forward(self, input):
        """from input[metapath_id][v_id][embedding] to [v_id][embedding]"""
        attention = torch.FloatTensor(input.size()[1], input.size()[0])  # shape(|vertices|,num_metapath)
        for mp_id in range(input.size()[0]):  # mp_id: metapath_idx
            attention[:, mp_id] = torch.mm(input[mp_id, :, :], self.W[:, mp_id])
        # attention = F.dropout(attention, self.dropout, training=self.training)
        attention = F.relu(attention)
        attention = F.softmax(attention, dim=1)
        # attention = F.dropout(attention, self.dropout, training=self.training)

        output = torch.FloatTensor(input.size()[1], input.size()[2])  # shape(|vertices|,embed_len)
        for v_id in range(input.size()[1]):
            for mp_id in range(input.size()[0]):
                output[v_id, :] = output[v_id, :] + input[mp_id, v_id, :] * attention[v_id, mp_id]

        if self.concat:
            return F.elu(output)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphInductiveAttention(Module):
    """
    Graph inductive attention layer, similar function to GCN. Attention+GraphSage
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphInductiveAttention, self).__init__()
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
