#!/usr/bin/env python

"""
    models.py
"""

from __future__ import division
from __future__ import print_function

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from lr import LRSchedule

from nn_modules import GraphConvolution

# --
# Model

class HINGCN_GS(nn.Module):
    def __init__(self,
                 n_mp,
                 problem,
                 prep_len,
                 n_head,
                 layer_specs,
                 aggregator_class,
                 mpaggr_class,
                 edgeupt_class,
                 prep_class,
                 sampler_class,
                 dropout,
                 batchnorm,
                 bias=False,
                 ):

        super(HINGCN_GS, self).__init__()

        # --
        # Input Data
        self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head
        self.bias = bias

        # self.feats
        self.register_buffer('feats', problem.feats)

        # self.edge_emb_mp
        for i,key in enumerate(problem.edge_emb):
            self.register_buffer('edge_emb_{}'.format(i), problem.edge_emb[key])
        # self.adjs_mp
        for i,key in enumerate(problem.adj):
            self.register_buffer('adjs_{}'.format(i), problem.adj[key])

        # Define network
        self.n_mp = n_mp
        self.depth = len(layer_specs)
        self.dropout = dropout
        self.batchnorm = batchnorm

        # Sampler
        self.train_sampler = sampler_class()
        self.val_sampler = sampler_class()
        self.train_sample_fns = [partial(self.train_sampler, n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [partial(self.val_sampler, n_samples=s['n_val_samples']) for s in layer_specs]

        # Prep
        self.prep = prep_class(input_dim=problem.feats_dim, n_nodes=problem.n_nodes,output_dim=prep_len, embedding_dim = prep_len)
        self.input_dim = self.prep.output_dim

        # Network
        for mp in range(self.n_mp):
            # agg_layers = []
            # edge_layers = []
            input_dim = self.input_dim
            for i, spec in enumerate(layer_specs):
                agg = nn.ModuleList( [ aggregator_class(
                    input_dim=input_dim,
                    edge_dim=problem.edge_dim,
                    output_dim=spec['output_dim'],
                    activation=spec['activation'],
                    concat_node=spec['concat_node'],
                    concat_edge=spec['concat_edge'],
                    dropout=self.dropout,
                    batchnorm=self.batchnorm,
                ) for _ in range(n_head) ])
                # agg_layers.append(agg)
                input_dim = agg[0].output_dim * n_head  # May not be the same as spec['output_dim']

                edge = edgeupt_class(
                    input_dim=input_dim,
                    edge_dim=self.edge_dim,
                    activation=spec['activation'],
                    dropout=self.dropout,
                    batchnorm=self.batchnorm,
                )
                # edge_layers.append(edge)

                self.add_module('agg_{}_{}'.format(mp, i), agg)
                self.add_module('edge_{}_{}'.format(mp, i), edge)
        if self.bias:
            self.n_homo_nodes = problem.homo_feat.shape[0]
            back_emb = nn.Embedding(self.n_homo_nodes-problem.n_nodes,
                                         prep_len)
            back_emb.from_pretrained(problem.homo_feat[problem.n_nodes+1:-1],freeze=False)
            self.add_module('back_emb', back_emb)

            self.background=nn.Sequential(*[
            GraphConvolution(prep_len, 64, adj=problem.homo_adj),
            nn.ReLU(), nn.Dropout(self.dropout),
            GraphConvolution(64, 32, adj=problem.homo_adj),
            nn.ReLU(), nn.Dropout(self.dropout),
            GraphConvolution(32, 16, adj=problem.homo_adj),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(16, input_dim),
        ])
        self.mp_agg = mpaggr_class(input_dim,n_head=self.n_mp+int(self.bias),dropout=self.dropout,batchnorm=self.batchnorm,)

        self.fc = nn.Linear(self.mp_agg.output_dim, problem.n_classes, bias=True)


    # We only want to forward IDs to facilitate nn.DataParallelism
    def forward(self, ids, train=True):

        # print("\tIn Model: input size ", ids.shape)
        # ids.to(self.feats.device)

        # Sample neighbors
        sample_fns = self.train_sample_fns if train else self.val_sample_fns

        has_feats = self.feats is not None

        output = []
        tmp_ids = ids
        for mp in range(self.n_mp):
            ids = tmp_ids
            tmp_feats = self.feats[ids] if has_feats else None
            all_feats = [self.prep(ids, tmp_feats, layer_idx=0)]
            all_edges = []
            all_masks = []
            for layer_idx, sampler_fn in enumerate(sample_fns):
                neigh, edges, mask = sampler_fn(adj=getattr(self,'adjs_{}'.format(mp)), ids=ids)

                all_edges.append(getattr(self,'edge_emb_{}'.format(mp))[edges.contiguous().view(-1)])
                all_masks.append(mask)

                ids = neigh.contiguous().view(-1)
                tmp_feats = self.feats[ids] if has_feats else None
                all_feats.append(self.prep(ids, tmp_feats, layer_idx=layer_idx + 1))

            # Sequentially apply layers, per original (little weird, IMO)
            # Each iteration reduces length of array by one
            for i in range(self.depth):
                all_feats = [ torch.cat([getattr(self,'agg_{}_{}'.format(mp, i))[h] (all_feats[k], all_feats[k + 1],
                                                     all_edges[k], mask=all_masks[k]) for h in range(self.n_head)], dim=1)
                              for k in range(len(all_feats) - 1)]
                all_feats = [F.dropout(i, self.dropout, training=self.training) for i in all_feats]
                all_edges = [getattr(self,'edge_{}_{}'.format(mp, i))(all_feats[k], all_feats[k + 1],
                                                    all_edges[k], mask=all_masks[k]
                                                    ) for k in range(len(all_edges) - 1)]
                all_edges = [F.dropout(i, self.dropout, training=self.training) for i in all_edges]

            assert len(all_feats) == 1, "len(all_feats) != 1"
            output.append(all_feats[0].unsqueeze(0))
        if self.bias:
            tmp_feats = self.feats[ids]
            all_feats = self.prep(torch.arange(self.n_nodes).to(ids.device), tmp_feats, layer_idx=1)
            back_ids = torch.arange(self.n_homo_nodes-self.n_nodes).to(ids.device)
            all_feats = torch.cat([all_feats,self.back_emb(back_ids)],dim=0)
            output.append(self.background(all_feats)[tmp_ids].unsqueeze(0))
        output = torch.cat(output)
        
        #output = F.normalize(output, dim=2) #normalize before attention

        output, weights = self.mp_agg(output)
        # print(weights)
        output = F.normalize(output, dim=1)  # ?? Do we actually want this? ... Sometimes ...
        output = F.dropout(output, self.dropout, training=self.training)
        output = (self.fc(output))
        return output, weights


class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        out =self.linear(x)
        return out



class HINGCN_Dense(nn.Module):
    def __init__(self,
                 n_mp,
                 problem,
                 prep_len,
                 n_head,
                 layer_specs,
                 aggregator_class,
                 mpaggr_class,
                 edgeupt_class,
                 prep_class,
                 sampler_class,
                 dropout,
                 batchnorm,
                 bias=False,
                 ):

        super(HINGCN_Dense, self).__init__()

        # --
        # Input Data
        self.edge_dim = problem.edge_dim
        self.input_dim = problem.feats_dim
        self.n_nodes = problem.n_nodes
        self.n_classes = problem.n_classes
        self.n_head = n_head
        self.bias = bias

        # self.feats
        self.register_buffer('feats', problem.feats)

        # self.edge_emb_mp
        for i,key in enumerate(problem.edge_emb):
            self.register_buffer('edge_emb_{}'.format(i), problem.edge_emb[key])
        # self.adjs_mp
        for i,key in enumerate(problem.adj):
            self.register_buffer('adjs_{}'.format(i), problem.adj[key])

        # Define network
        self.n_mp = n_mp
        self.depth = len(layer_specs)
        self.dropout = dropout
        self.batchnorm = batchnorm

        # Sampler
        self.train_sampler = sampler_class()
        self.val_sampler = sampler_class()
        self.train_sample_fns = [partial(self.train_sampler, n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [partial(self.val_sampler, n_samples=s['n_val_samples']) for s in layer_specs]

        # Prep
        self.prep = prep_class(input_dim=problem.feats_dim, n_nodes=problem.n_nodes, embedding_dim = prep_len)
        self.input_dim = self.prep.output_dim

        # Network
        for mp in range(self.n_mp):
            # agg_layers = []
            # edge_layers = []
            input_dim = self.input_dim
            for i, spec in enumerate(layer_specs):
                agg = nn.ModuleList( [ aggregator_class(
                    input_dim=input_dim,
                    edge_dim=problem.edge_dim,
                    output_dim=spec['output_dim'],
                    activation=spec['activation'],
                    concat_node=spec['concat_node'],
                    concat_edge=spec['concat_edge'],
                    dropout=self.dropout,
                    batchnorm=self.batchnorm,
                ) for _ in range(n_head) ])
                # agg_layers.append(agg)
                input_dim = agg[0].output_dim * n_head  # May not be the same as spec['output_dim']

                self.add_module('agg_{}_{}'.format(mp, i), agg)
        # if self.bias:
        #     self.n_homo_nodes = problem.homo_feat.shape[0]
        #     back_emb = nn.Embedding(self.n_homo_nodes-problem.n_nodes,
        #                                  prep_len)
        #     back_emb.from_pretrained(problem.homo_feat[problem.n_nodes+1:-1],freeze=False)
        #     self.add_module('back_emb', back_emb)
        #
        #     self.background=nn.Sequential(*[
        #     GraphConvolution(problem.homo_feat.shape[1], 64, adj=problem.homo_adj),
        #     nn.ReLU(), nn.Dropout(self.dropout),
        #     GraphConvolution(64, 32, adj=problem.homo_adj),
        #     nn.ReLU(), nn.Dropout(self.dropout),
        #     GraphConvolution(32, 16, adj=problem.homo_adj),
        #     nn.ReLU(), nn.Dropout(self.dropout),
        #     nn.Linear(16, input_dim),
        # ])
        self.mp_agg = mpaggr_class(input_dim,n_head=self.n_mp, dropout=self.dropout,batchnorm=self.batchnorm,)

        self.fc = nn.Linear(self.mp_agg.output_dim, problem.n_classes, bias=True)


    # We only want to forward IDs to facilitate nn.DataParallelism
    def forward(self, ids, train=True):
        # Sample neighbors
        sample_fns = self.train_sample_fns if train else self.val_sample_fns

        has_feats = self.feats is not None

        output = []
        tmp_ids = ids
        for mp in range(self.n_mp):
            ids = tmp_ids
            tmp_feats = self.feats[ids] if has_feats else None
            all_feats = [self.prep(ids, tmp_feats, layer_idx=0)]
            all_edges = []
            all_masks = []
            for layer_idx, sampler_fn in enumerate(sample_fns):
                neigh, edges, mask = sampler_fn(adj=getattr(self,'adjs_{}'.format(mp)), ids=ids)

                all_edges.append([])
                all_masks.append(mask)

                ids = torch.arange(self.n_nodes).to(tmp_ids.device)
                tmp_feats = self.feats[ids] if has_feats else None
                all_feats.append(self.prep(ids, tmp_feats, layer_idx=layer_idx + 1))

            # Sequentially apply layers, per original (little weird, IMO)
            # Each iteration reduces length of array by one
            for i in range(self.depth):
                all_feats = [ torch.cat([getattr(self,'agg_{}_{}'.format(mp, i))[h] (all_feats[k], all_feats[k + 1],
                                                     all_edges[k], mask=all_masks[k]) for h in range(self.n_head)], dim=1)
                              for k in range(len(all_feats) - 1)]
                all_feats = [F.dropout(i, self.dropout, training=self.training) for i in all_feats]

            output.append(all_feats[0].unsqueeze(0))
        # if self.bias:
        #     tmp_feats = None
        #     all_feats = self.prep(torch.arange(self.n_nodes).to(ids.device), tmp_feats, layer_idx=1)
        #     back_ids = torch.arange(self.n_homo_nodes-self.n_nodes).to(ids.device)
        #     all_feats = torch.cat([all_feats,self.back_emb(back_ids)],dim=0)
        #     output.append(self.background(all_feats)[tmp_ids].unsqueeze(0))
        output = torch.cat(output)

        output, weights = self.mp_agg(output)
        # print(weights)
        # output = F.normalize(output, dim=1)  # ?? Do we actually want this? ... Sometimes ...
        output = F.dropout(output, self.dropout, training=self.training)

        return self.fc(output), weights
