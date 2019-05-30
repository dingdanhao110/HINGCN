#!/usr/bin/env python

"""
    problem.py
"""

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import sparse
from sklearn import metrics

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from helpers import load_edge_emb

from helpers import read_mpindex_dblp,sparse_mx_to_torch_sparse_tensor

# --
# Helper classes

class ProblemLosses:
    @staticmethod
    def multilabel_classification(preds, targets):
        return F.multilabel_soft_margin_loss(preds, targets)
    
    @staticmethod
    def classification(preds, targets):
        return F.cross_entropy(preds, targets)
        
    @staticmethod
    def regression_mae(preds, targets):
        return F.l1_loss(preds, targets)
        
    # @staticmethod
    # def regression_mse(preds, targets):
    #     return F.mse_loss(preds - targets)


class ProblemMetrics:
    @staticmethod
    def multilabel_classification(y_true, y_pred):
        y_pred = (y_pred > 0).astype(int)
        return {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "micro" : float(metrics.f1_score(y_true, y_pred, average="micro")),
            "macro" : float(metrics.f1_score(y_true, y_pred, average="macro")),
        }
    
    @staticmethod
    def classification(y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        return {
            "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
            "micro" : float(metrics.f1_score(y_true, y_pred, average="micro")),
            "macro" : float(metrics.f1_score(y_true, y_pred, average="macro")),
        }
        # return (y_pred == y_true.squeeze()).mean()
    
    @staticmethod
    def regression_mae(y_true, y_pred):
        return float(np.abs(y_true - y_pred).mean())


# --
# Problem definition

class NodeProblem(object):
    def __init__(self, problem_path, schemes, cuda=True):
        
        print('NodeProblem: loading started')

        features, labels, folds = read_mpindex_dblp(path=problem_path)

        edge_index, edge_emb = load_edge_emb(path=problem_path,
                                             schemes=schemes,
                                             n_dim=16)

        self.task      = 'classification'
        self.n_classes = 4 # !!

        #input: features, homograph, edge embedding
        self.feats = features
        self.adj = edge_index
        self.edge_emb = edge_emb

        self.schemes=schemes

        self.folds     = folds
        self.targets   = labels

        self.feats_dim = self.feats.shape[1] if self.feats is not None else None
        self.edge_dim = edge_emb[schemes[0]].shape[1]
        self.n_nodes   = self.adj[schemes[0]].shape[0]
        self.cuda      = cuda
        self.__to_torch()
        
        self.nodes = {
            "train" : self.folds ['train'],
            "val"   : self.folds ['val'],
            "test"  : self.folds ['test'],
        }
        
        self.loss_fn = getattr(ProblemLosses, self.task)
        self.metric_fn = getattr(ProblemMetrics, self.task)
        
        print('NodeProblem: loading finished')
    
    def __to_torch(self):
        self.feats = torch.FloatTensor(self.feats)

        # if not sparse.issparse(self.adj):
        if self.cuda:
                for i in self.adj:
                    self.adj[i]=self.adj[i].cuda()
                    print(torch.cuda.memory_allocated())
                for i in self.edge_emb:
                    if torch.is_tensor(self.edge_emb[i]):
                        self.edge_emb[i] = self.edge_emb[i].cuda()
                    print(torch.cuda.memory_allocated())

        if self.feats is not None:
            if self.cuda:
                self.feats = self.feats.cuda()
                print(torch.cuda.memory_allocated())

    def __batch_to_torch(self, mids, targets):
        """ convert batch to torch """
        mids = Variable(torch.LongTensor(mids))
        
        if self.task == 'multilabel_classification':
            targets = Variable(torch.FloatTensor(targets))
        elif self.task == 'classification':
            targets = Variable(torch.LongTensor(targets))
        elif 'regression' in self.task:
            targets = Variable(torch.FloatTensor(targets))
        else:
            raise Exception('NodeDataLoader: unknown task: %s' % self.task)
        
        if self.cuda:
            mids, targets = mids.cuda(), targets.cuda()
        
        return mids, targets
    
    def iterate(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]
        
        idx = np.arange(nodes.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)
        
        n_chunks = idx.shape[0] // batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            mids = nodes[chunk]
            targets = self.targets[mids].reshape(-1,1)
            mids, targets = self.__batch_to_torch(mids, targets)
            yield mids, targets, chunk_id / n_chunks
