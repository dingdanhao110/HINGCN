import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *
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
        self.attention = MetapathAggrLayer(dim_mp, nmeta, dropout=dropout, alpha=alpha, concat=bias)
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
    def __init__(self, nfeat, nhid, nmeta, dim_mp, nclass, alpha, dropout, bias
                 , sampler, adjs, concat, samples=128
                 ):
        super(HINGCN_IA, self).__init__()
        self.adjs = adjs
        self.nmeta = nmeta
        self.nsamples = samples
        self.concat = concat
        self.dropout = dropout

        self.aggr_layer1 = [AttentionAggregator(nfeat, nhid, dropout=dropout, alpha=alpha, concat=concat) for _ in range(nmeta)]
        for i, gcn in enumerate(self.aggr_layer1):
            self.add_module('gcn_1_{}'.format(i), gcn)

        if concat:
            nhid*=2

        self.aggr_layer2 = [AttentionAggregator(nhid, dim_mp, dropout=dropout, alpha=alpha, concat=concat) for _ in range(nmeta)]
        for i, gcn in enumerate(self.aggr_layer2):
            self.add_module('gcn_2_{}'.format(i), gcn)

        if concat:
            dim_mp*=2

        self.sampler_layer1 = [sampler(adjs[i]) for i in range(nmeta)]
        # for i, sam in enumerate(self.sampler_layer1):
        #     self.add_module('sampler_1_{}'.format(i), sam)

        self.sampler_layer2 = [sampler(adjs[i]) for i in range(nmeta)]
        # for i, sam in enumerate(self.sampler_layer2):
        #     self.add_module('sampler_2_{}'.format(i), sam)

        self.attention = MetapathAggrLayer(dim_mp, nmeta, dropout=dropout, alpha=alpha)
        self.linear = nn.Linear(dim_mp,nclass,bias=bias)


    def forward(self, input):
        """
        Args:
            input: feature matrix of queried vertices in HIN;
        Return:
            logits of classification for queried vertices in HIN
        """
        ids = np.arange(input.shape[0])
        embeddings = []
        for mp_idx in range(self.nmeta):
            neibor_i_1 = self.sampler_layer1[mp_idx](ids, self.nsamples)

            x_i = F.relu(self.aggr_layer1[mp_idx](input, neibor_i_1))

            x_i = F.dropout(x_i, self.dropout, training=self.training)
            neibor_i_2 = self.sampler_layer2[mp_idx](ids, self.nsamples)

            x_i = F.relu(self.aggr_layer2[mp_idx](x_i, neibor_i_2))  #x_i: tensor(v_id,embedding)
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            embeddings.append(x_i.unsqueeze(0))
        embeddings = torch.cat(embeddings)  #should be a tensor of (mp_idx,v_id,embedding)

        output = self.attention(embeddings)
        output = F.dropout(output, self.dropout, training=self.training)
        output = F.relu(self.linear(output))
        return F.log_softmax(output, dim=1)



class HINGCN_edge(nn.Module):
    """replaced GCN layers by inductive attention layers"""
    def __init__(self, nfeat, nhid, nmeta, dim_mp,
                 edge_dim, schemes, nclass, alpha, dropout, bias,
                 adjs, concat, samples=128
                 ):
        super(HINGCN_edge, self).__init__()
        self.adjs = adjs
        self.nmeta = nmeta
        self.nsamples = samples
        self.concat = concat
        self.dropout = dropout
        self.edge_dim=edge_dim
        self.schemes=schemes

        assert len(schemes)==nmeta

        # TODO: check dimension
        self.aggr_layer1 = [EdgeAttentionAggregator(nfeat, nhid, edge_dim, schemes[i],
                                                    dropout=dropout, alpha=alpha, concat=concat) for i in range(nmeta)]
        for i, gcn in enumerate(self.aggr_layer1):
            self.add_module('gcn_1_{}'.format(i), gcn)

        if concat:
            inter_len = nhid*2+edge_dim
        else:
            inter_len = nhid*2
        self.aggr_layer2 = [EdgeAttentionAggregator(inter_len, dim_mp, edge_dim, schemes[i],
                                                    dropout=dropout, alpha=alpha, concat=concat) for i in range(nmeta)]
        for i, gcn in enumerate(self.aggr_layer2):
            self.add_module('gcn_2_{}'.format(i), gcn)


        if concat:
            res_len = dim_mp*2+edge_dim
        else:
            res_len = dim_mp*2

        self.attention = MetapathAggrLayer(res_len, nmeta, dropout=dropout, alpha=alpha)
        self.linear = nn.Linear(res_len,nclass,bias=bias)


    def forward(self, input, index, node_emb, n_sample=128):
        """
        Args:
            input: feature matrix of queried vertices in HIN;
        Return:
            logits of classification for queried vertices in HIN
        """
        # ids = np.arange(input.shape[0])
        embeddings = []
        for mp_idx in range(self.nmeta):
            x_i = F.relu(self.aggr_layer1[mp_idx](input, index, node_emb, n_sample))
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            x_i = F.relu(self.aggr_layer2[mp_idx](x_i, index, node_emb, n_sample))  #x_i: tensor(v_id,embedding)
            x_i = F.dropout(x_i, self.dropout, training=self.training)
            embeddings.append(x_i.unsqueeze(0))
        embeddings = torch.cat(embeddings)  #should be a tensor of (mp_idx,v_id,embedding)

        output = self.attention(embeddings)
        output = F.dropout(output, self.dropout, training=self.training)
        output = F.relu(self.linear(output))
        return F.log_softmax(output, dim=1)