import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, GraphAttentionLayer, MetapathAttentionLayer
from inductive_modules import *


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)


class HINGCN(nn.Module):
    def __init__(self, nfeat, nhid, nmeta, dim_mp, nclass, alpha, dropout, bias):
        super(HINGCN, self).__init__()

        self.gcn_layer1 = [GraphConvolution(nfeat, nhid) for _ in range(nmeta)]
        for i, gcn in enumerate(self.gcn_layer1):
            self.add_module('gcn_1_{}'.format(i), gcn)

        self.gcn_layer2 = [GraphConvolution(nhid, dim_mp) for _ in range(nmeta)]
        for i, gcn in enumerate(self.gcn_layer2):
            self.add_module('gcn_2_{}'.format(i), gcn)

        self.dropout = dropout
        self.attention = MetapathAttentionLayer(dim_mp, nmeta, dropout=dropout, alpha=alpha, concat=bias)
        self.linear = nn.Linear(dim_mp,nclass,bias=bias)


    def forward(self, input, adjs):
        """@:param input: feature matrix of queried vertices in HIN;
                   adjs: tensor of homogeneous adj matrices; adjs[mp_idx,v1,v2];
           @:return logits of classification for queried vertices in HIN
        """
        embeddings = []
        for mp_idx in range(len(adjs)):
            x_i = F.relu(self.gcn_layer1[mp_idx](input, adjs[mp_idx]))
            # x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = F.relu(self.gcn_layer2[mp_idx](x_i, adjs[mp_idx]))  #x_i: tensor(v_id,embedding)
            # x_i = F.dropout(x_i, self.dropout, training=self.training)
            embeddings.append(x_i.unsqueeze(0))
        embeddings = torch.cat(embeddings)  #should be a tensor of (mp_idx,v_id,embedding)

        output = self.attention(embeddings)
        # output = F.dropout(output, self.dropout, training=self.training)
        output = F.relu(self.linear(output))
        return F.log_softmax(output, dim=1)



class HINGCN_IA(nn.Module):
    """replaced GCN layers by inductive attention layers"""
    def __init__(self, nfeat, nhid, nmeta, dim_mp, nclass, alpha, dropout, bias):
        super(HINGCN_IA, self).__init__()

        self.aggr_layer1 = [AttentionAggregator(nfeat, nhid, dropout=dropout, alpha=alpha, concat=bias) for _ in range(nmeta)]
        for i, gcn in enumerate(self.aggr_layer1):
            self.add_module('gcn_1_{}'.format(i), gcn)

        self.aggr_layer2 = [AttentionAggregator(nhid, dim_mp, dropout=dropout, alpha=alpha, concat=bias) for _ in range(nmeta)]
        for i, gcn in enumerate(self.aggr_layer2):
            self.add_module('gcn_2_{}'.format(i), gcn)

        self.dropout = dropout
        self.attention = MetapathAttentionLayer(dim_mp, nmeta, dropout=dropout, alpha=alpha, concat=bias)
        self.linear = nn.Linear(dim_mp,nclass,bias=bias)


    def forward(self, input, adjs):
        """@:param input: feature matrix of queried vertices in HIN;
                   adjs: list of tensor of homogeneous adj matrices; adjs[mp_idx][v1,v2];
           @:return logits of classification for queried vertices in HIN
        """
        embeddings = []
        for mp_idx in range(len(adjs)):
            neibor_i_1 = UniformNeighborSampler(adjs[mp_idx])
            x_i = F.relu(self.aggr_layer1[mp_idx](input, neibor_i_1))
            # x_i = F.dropout(x_i, self.dropout, training=self.training)
            neibor_i_2 = UniformNeighborSampler(adjs[mp_idx])
            x_i = F.relu(self.aggr_layer2[mp_idx](x_i, neibor_i_2))  #x_i: tensor(v_id,embedding)
            # x_i = F.dropout(x_i, self.dropout, training=self.training)
            embeddings.append(x_i.unsqueeze(0))
        embeddings = torch.cat(embeddings)  #should be a tensor of (mp_idx,v_id,embedding)

        output = self.attention(embeddings)
        # output = F.dropout(output, self.dropout, training=self.training)
        output = F.relu(self.linear(output))
        return F.log_softmax(output, dim=1)