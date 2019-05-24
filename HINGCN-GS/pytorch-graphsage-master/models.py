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
                 schemes,
                 lr_init=0.01,
                 weight_decay=0.0,
                 lr_schedule='constant',
                 epochs=10):

        super(HINGCN_GS, self).__init__()

        # --
        # Define network
        self.schemes = schemes

        # Sampler
        self.train_sampler = sampler_class()
        self.val_sampler = sampler_class()
        self.train_sample_fns = [partial(self.train_sampler, n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [partial(self.val_sampler, n_samples=s['n_val_samples']) for s in layer_specs]

        # Prep
        self.prep = prep_class(input_dim=input_dim, n_nodes=n_nodes)
        input_dim = self.prep.output_dim

        # Network
        self.para_layers = []
        self.emb_layers = []
        for mp in range(len(schemes)):
            agg_layers = []
            edge_layers = []
            for spec in enumerate(layer_specs):
                agg = aggregator_class(
                    input_dim=input_dim,
                    output_dim=spec['output_dim'],
                    activation=spec['activation'],
                )
                agg_layers.append(agg)
                input_dim = agg.output_dim  # May not be the same as spec['output_dim']
                edge_layers.append(edgeupt_class(input_dim, edge_dim))

            self.para_layers.append(torch.nn.Sequential(*agg_layers))
            self.emb_layers.append(torch.nn.Sequential(*edge_layers))

        self.mp_agg = mpaggr_class(input_dim)

        self.fc = nn.Linear(input_dim, n_classes, bias=True)

        # --
        # Define optimizer

        self.lr_scheduler = partial(getattr(LRSchedule, lr_schedule), lr_init=lr_init)
        self.lr = self.lr_scheduler(0.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)

    def forward(self, ids, feats, adjs, edge_emb, train=True):
        # Sample neighbors
        sample_fns = self.train_sample_fns if train else self.val_sample_fns

        has_feats = feats is not None

        output=[]
        for mp in range(len(self.schemes)):

            tmp_feats = feats[ids] if has_feats else None
            all_feats = [self.prep(ids, tmp_feats, layer_idx=0)]
            all_edges = []
            for layer_idx, sampler_fn in enumerate(sample_fns):
                neigh = sampler_fn(adj=adjs, ids=ids).contiguous().view(-1)
                n_sample = neigh.shape[0] / ids.shape[0]
                all_edges.append(edge_emb[adjs[
                    ids.view(-1, 1).repeat(1, n_sample).view(-1),
                    neigh]])

                ids = neigh
                tmp_feats = feats[ids] if has_feats else None
                all_feats.append(self.prep(ids, tmp_feats, layer_idx=layer_idx + 1))

            # Sequentially apply layers, per original (little weird, IMO)
            # Each iteration reduces length of array by one
            for i in range(len(self.para_layers[mp])):
                all_feats = [self.para_layers[mp][i](all_feats[k], all_feats[k + 1],
                                       all_edges[k]
                                       ) for k in range(len(all_feats) - 1)]
                all_edges = [self.emb_layers[mp][i](all_feats[k], all_feats[k + 1],
                                       all_edges[k]
                                       ) for k in range(len(all_edges) - 1)]

            assert len(all_feats) == 1, "len(all_feats) != 1"
            output.append(all_feats[0].unsqueeze(0))
        output = torch.cat(output)
        output = self.mp_agg(output)
        # out = F.normalize(output, dim=1)  # ?? Do we actually want this? ... Sometimes ...
        return F.relu(self.fc(output))

    def set_progress(self, progress):
        self.lr = self.lr_scheduler(progress)
        LRSchedule.set_lr(self.optimizer, self.lr)

    def train_step(self, ids, feats, targets, loss_fn):
        self.optimizer.zero_grad()
        preds = self(ids, feats, train=True)
        loss = loss_fn(preds, targets.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return preds
