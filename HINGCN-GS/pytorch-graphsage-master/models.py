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


# --
# Model

class HINGCN_GS(nn.Module):
    def __init__(self,
                 input_dim,
                 edge_dim,
                 n_nodes,
                 n_classes,
                 layer_specs,
                 aggregator_class,
                 mpaggr_class,
                 edgeupt_class,
                 prep_class,
                 sampler_class,
                 n_mp,
                 problem,
                 lr_init=0.01,
                 weight_decay=0.0,
                 lr_schedule='constant',
                 dropout=0.5,
                 ):

        super(HINGCN_GS, self).__init__()

        # --
        # Graph Data
        # self.feats = problem.feats
        # self.edge_emb = problem.edge_emb
        # self.adjs = problem.adj

        self.register_buffer('feats', problem.feats)

        for i,key in enumerate(problem.edge_emb):
            self.register_buffer('edge_emb_{}'.format(i), problem.edge_emb[key])
        for i,key in enumerate(problem.adj):
            self.register_buffer('adjs_{}'.format(i), problem.adj[key])

        # Define network
        self.n_mp = n_mp
        # self.register_buffer('schemes', schemes)
        self.dropout = dropout

        # Sampler
        self.train_sampler = sampler_class()
        self.val_sampler = sampler_class()
        self.train_sample_fns = [partial(self.train_sampler, n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [partial(self.val_sampler, n_samples=s['n_val_samples']) for s in layer_specs]

        # Prep
        self.prep = prep_class(input_dim=input_dim, n_nodes=n_nodes)
        self.input_dim = self.prep.output_dim

        # Network
        self.para_layers = []
        self.emb_layers = []
        for mp in range(self.n_mp):
            agg_layers = []
            edge_layers = []
            input_dim = self.input_dim
            for i, spec in enumerate(layer_specs):
                agg = aggregator_class(
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    output_dim=spec['output_dim'],
                    activation=spec['activation'],
                    concat_node=spec['concat_node'],
                    concat_edge=spec['concat_edge'],
                    dropout=self.dropout,
                )
                agg_layers.append(agg)
                input_dim = agg.output_dim  # May not be the same as spec['output_dim']

                edge = edgeupt_class(
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    activation=spec['activation'],
                )
                edge_layers.append(edge)

                self.add_module('agg_{}_{}'.format(mp, i), agg)
                self.add_module('edge_{}_{}'.format(mp, i), edge)

            self.para_layers.append(agg_layers)
            self.emb_layers.append(edge_layers)

        self.mp_agg = mpaggr_class(input_dim)

        self.fc = nn.Linear(input_dim, n_classes, bias=True)

        # --
        # Define optimizer

        self.lr_scheduler = partial(getattr(LRSchedule, lr_schedule), lr_init=lr_init)
        self.lr = self.lr_scheduler(0.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

    # We only want to forward IDs to facilitate nn.DataParallelism
    def forward(self, ids, train=True):

        print("\tIn Model: input size ", ids.shape)

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
            for layer_idx, sampler_fn in enumerate(sample_fns):
                neigh, edges = sampler_fn(adj=getattr(self,'adjs_{}'.format(mp)), ids=ids)

                all_edges.append(getattr(self,'edge_emb_{}'.format(mp))[edges.contiguous().view(-1)])

                ids = neigh.contiguous().view(-1)
                tmp_feats = self.feats[ids] if has_feats else None
                all_feats.append(self.prep(ids, tmp_feats, layer_idx=layer_idx + 1))

            # Sequentially apply layers, per original (little weird, IMO)
            # Each iteration reduces length of array by one
            for i in range(len(self.para_layers[mp])):
                all_feats = [self.para_layers[mp][i](all_feats[k], all_feats[k + 1],
                                                     all_edges[k]
                                                     ) for k in range(len(all_feats) - 1)]
                all_feats = [F.dropout(i, self.dropout, training=self.training) for i in all_feats]
                all_edges = [self.emb_layers[mp][i](all_feats[k], all_feats[k + 1],
                                                    all_edges[k]
                                                    ) for k in range(len(all_edges) - 1)]
                all_edges = [F.dropout(i, self.dropout, training=self.training) for i in all_edges]

            assert len(all_feats) == 1, "len(all_feats) != 1"
            output.append(all_feats[0].unsqueeze(0))
        output = torch.cat(output)
        output = self.mp_agg(output)
        output = F.normalize(output, dim=1)  # ?? Do we actually want this? ... Sometimes ...
        output = F.dropout(output, self.dropout, training=self.training)

        return self.fc(output)



class MyDataParallel(nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
