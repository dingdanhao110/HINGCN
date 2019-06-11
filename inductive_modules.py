import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from scipy import sparse
from utilities import sparse_mx_to_torch_sparse_tensor, normalize
from metapath import query_path, query_path_indexed


def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)

    return x.cpu().numpy() if x.is_cuda else x.numpy()


def adj_list_to_adj_mat(neigh):
    """from dense adj list neigh to adj mat"""
    # tmp = np.zeros((neigh.shape[0],neigh.shape[0]),dtype=bool)

    tmp = sparse.coo_matrix((np.ones(neigh.size),
                             (np.arange(neigh.shape[0]).repeat(neigh.shape[1]).reshape(-1),
                              np.array(neigh).reshape(-1))))

    return tmp


# --
# Samplers
def UniformNeighborSampler(adj, n_samples=128):
    """
        Samples from "sparse 2D edgelist" COO matrix, according to adj. adj[v1,v2]=1: connected

        :return sparse neighbor adj mat
    """
    assert n_samples > 0, 'UniformNeighborSampler: n_samples > 0'

    adj_np = adj.numpy().to_csr()
    degrees = np.count_nonzero(adj_np, axis=1)
    degrees[degrees == 0] = adj_np.shape[1]  # if no degree at all, sample from all vertices

    sel = np.random.choice(adj_np.shape[1], (adj_np.shape[0], n_samples))
    sel = sel % degrees.reshape(-1, 1)

    nonzeros = np.split(adj_np.indices, adj_np.indptr)[1:-1]  ##nonzeros for each row
    nonzeros[degrees == adj_np.shape[1]] = np.arange(0,
                                                     adj_np.shape[0])  ##if no degree at all, sample from all vertices

    tmp = nonzeros[np.arange(adj_np.shape[0]).repeat(n_samples).reshape(-1),
                   np.array(sel).reshape(-1)]

    tmp = adj_list_to_adj_mat(tmp)
    tmp = sparse_mx_to_torch_sparse_tensor(tmp)

    if adj.is_cuda():
        tmp = tmp.cuda()

    return tmp


class WeightedNeighborSampler():
    """
        weighted sampling from "sparse 2D edgelist" COO matrix.

    """

    def __init__(self, adj):
        assert adj.is_sparse, "WeightedNeighborSampler: not sparse.issparse(adj)"
        self.is_cuda = adj.is_cuda
        self.adj = normalize(adj.to_dense().numpy())
        self.degrees = np.count_nonzero(self.adj, axis=1)

    def __call__(self, ids, n_samples=128):
        assert n_samples > 0, 'WeightedNeighborSampler: n_samples must be set explicitly'

        sel = [np.random.choice(self.adj.shape[1], n_samples, p=self.adj[id], replace=False)
               if self.degrees[id] >= n_samples
               else np.random.choice(self.adj.shape[1], n_samples, p=self.adj[id], replace=True)
               for id in ids]
        sel = np.asarray(sel)
        tmp = Variable(torch.LongTensor(sel))

        if self.is_cuda:
            tmp = tmp.cuda()
        return tmp


sampler_lookup = {
    "uniform_neighbor_sampler": UniformNeighborSampler,
    "weighted_neighbor_sampler": WeightedNeighborSampler,
}


# --
# Aggregators

class AttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, alpha, concat=True):
        super(AttentionAggregator, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.output_dim = output_dim
        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, x, neibs):
        # Compute attention weights
        N = x.size()[0]
        n_sample = neibs.shape[1]
        x = torch.mm(x, self.W)
        a_input = torch.cat([x.repeat(1, n_sample).view(N * n_sample, -1),
                             x[neibs].view(N * n_sample, -1)], dim=1) \
            .view(N, -1, 2 * self.output_dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # e[ver,sample] attention coeff

        # Weighted average of neighbors
        attention = F.softmax(e, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        # h_prime = torch.matmul(attention, x[neibs])
        # h_prime = [ h_prime[id,id].unsqueeze(0) for id in range(N)]
        # h_prime = torch.cat(h_prime)

        h_prime = [torch.matmul(attention[i], x[neibs[i]]).unsqueeze(0) for i in range(N)]
        h_prime = torch.cat(h_prime)

        if self.concat:
            output = torch.cat([x, h_prime], dim=1)
        else:
            output = x + h_prime

        return F.elu(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class MetapathAggrLayer(nn.Module):
    """
    metapath attention layer.
    """

    def __init__(self, in_features, nmeta, dropout, alpha):
        super(MetapathAggrLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.alpha = alpha
        self.n_meta = nmeta
        # Weight: [in_features][1]
        self.a = nn.Parameter(torch.zeros(size=(in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input):
        # input: tensor(nmeta,N,in_features)
        input = input.transpose(0, 1)  # tensor(N,nmeta,in_features)
        N = input.size()[0]

        # a_input = torch.cat([input.repeat(1,1,self.nmeta).view(N, self.nmeta*self.nmeta, -1),
        #                      input.repeat(1,self.nmeta, 1)], dim=2).view(N, -1, 2 * self.in_features)
        a_input = input
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        e = F.softmax(e, dim=1)

        output = [torch.matmul(e[i], input[i]).unsqueeze(0) for i in range(N)]
        output = torch.cat(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class EdgeAttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, scheme, dropout, alpha, concat=True):
        super(EdgeAttentionAggregator, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.output_dim = output_dim
        self.scheme = scheme

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim + edge_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, features, index, node_emb, n_sample=128):
        # Compute attention weights
        N = features.size()[0]

        x = torch.mm(features, self.W)

        output = []
        for v in range(N):
            # generate neighbors of v
            neigh, emb = query_path_indexed(v, self.scheme, index, node_emb, n_sample)
            # assert neigh.shape[0] == n_sample
            n_neigh = neigh.shape[0]
            a_input = torch.cat([x[v].repeat(1, n_neigh).view(n_neigh, -1),
                                 x[neigh], emb], dim=1) \
                .view(n_neigh, -1)
            e = self.leakyrelu(torch.matmul(a_input, self.a).view(1, -1))
            attention = F.softmax(e, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            if self.concat:
                h_prime = torch.matmul(attention, torch.cat([x[neigh], emb], dim=1))
            else:
                h_prime = torch.matmul(attention, x[neigh])

            output.append(torch.cat([x[v], h_prime.squeeze()]))
        output = torch.stack(output)

        return F.elu(output)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


# use edge emb instead of query_path
class EdgeEmbAttentionAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, edge_dim, dropout, alpha,
                 concat=False, addedge=False, update_edge=False):
        super(EdgeEmbAttentionAggregator, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        self.output_dim = output_dim
        self.addedge = addedge
        self.update_edge = update_edge

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        self.W = nn.Parameter(torch.zeros(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.W2 = nn.Parameter(torch.zeros(size=(output_dim, edge_dim)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)

        self.W3 = nn.Parameter(torch.zeros(size=(edge_dim, edge_dim)))
        nn.init.xavier_uniform_(self.W3.data, gain=1.414)

        self.B = nn.Parameter(torch.zeros(size=(1, edge_dim)))
        nn.init.xavier_uniform_(self.B.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * output_dim + edge_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, features, index, node_emb, edge_index, edge_emb, n_sample=128):
        emb = edge_emb
        e_index = edge_index

        # Compute attention weights
        N = features.size()[0]
        x = torch.mm(features, self.W)

        # vectorize: each vertex sample n_sample neighbors;
        neigh = []
        for v in range(N):
            nonz = torch.nonzero(e_index[v]).view(-1)
            if (len(nonz) == 0):
                # no neighbor, only sample from itself
                # for edge embedding... PADDING with all-zero embedding at edge_emb[0]
                neigh.append(torch.LongTensor([v]).repeat(n_sample))
            else:
                idx = np.random.choice(nonz.shape[0], n_sample)
                neigh.append(nonz[idx])
        neigh = torch.stack(neigh).long()

        a_input = torch.cat([x.repeat(1, n_sample).view(N, n_sample, -1),
                             x[neigh],
                             emb[e_index[
                                 torch.arange(N).view(-1, 1).repeat(1, n_sample).view(-1),
                                 neigh.view(-1)]
                             ].view(N, n_sample, -1)], dim=2) \
            .view(N, n_sample, -1)

        e = self.leakyrelu(torch.matmul(a_input, self.a))
        attention = F.softmax(e, dim=1)
        attention = attention.squeeze(2)
        attention = F.dropout(attention, self.dropout, training=self.training)

        h_prime = [torch.matmul(attention[i], x[neigh[i]]) for i in range(N)]
        h_prime = torch.stack(h_prime)
        if self.concat:
            output = torch.cat([x, h_prime], dim=1)
        else:
            output = h_prime + x
        if self.addedge:
            output = torch.cat([output, torch.stack([
                torch.matmul(attention[i],
                             emb[e_index[i, neigh[i]]]) for i in range(N)])
                                ], dim=1)
        output = F.elu(output)

        # update edge
        if self.update_edge:
            to_update = e_index.nonzero()
            to_update = to_update[(to_update[:, 0] < to_update[:, 1]).nonzero().squeeze()]

            n = to_update.shape[0]

            #memory error.. consider minibatch update

            edges = e_index[to_update[:, 0], to_update[:, 1]]

            v_input = output[to_update]
            v_input = torch.matmul(v_input, self.W2).sum(dim=1)

            e_input = torch.mm(emb[edges], self.W3)

            a_input = torch.cat([self.B, e_input+v_input+self.B.repeat(n,1)],dim=0)

            emb = F.relu(a_input * emb)

        return output, emb

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


aggregator_lookup = {
    # "mean": MeanAggregator,
    # "max_pool": MaxPoolAggregator,
    # "mean_pool": MeanPoolAggregator,
    # "lstm": LSTMAggregator,
    "attention": AttentionAggregator,
    "eged_attention": EdgeAttentionAggregator,
    "edge_emb_attn": EdgeEmbAttentionAggregator,
}



class NodeEmbeddingPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, embedding_dim=64):
        """ adds node embedding """
        super(NodeEmbeddingPrep, self).__init__()

        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim)  # Affine transform, for changing scale + location

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
        if self.input_dim:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs
