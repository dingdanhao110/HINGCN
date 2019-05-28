#!/usr/bin/env python

"""
    helpers.py
"""

from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

def set_seeds(seed=0):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    if torch.cuda.is_available():
        _ = torch.cuda.manual_seed(seed)


def to_numpy(x):
    if isinstance(x, Variable):
        return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()


def read_embed(path="./data/dblp/",
               emb_file="APC"):
    with open("{}{}.emb".format(path, emb_file)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    embedding = np.loadtxt("{}{}.emb".format(path, emb_file),
                           dtype=np.float32, skiprows=1)
    emb_index = {}
    for i in range(n_nodes):
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:] for i in range(n_nodes)])

    assert features.shape[1] == n_feature
    assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature


def load_2hop_index(path="./data/dblp/", file="APA"):
    index = {}
    with open("{}{}.ind".format(path, file), mode='r') as f:
        for line in f:
            array = [int(x) for x in line.split()]
            a1 = array[0]
            a2 = array[1]
            if a1 not in index:
                index[a1] = {}
            if a2 not in index[a1]:
                index[a1][a2] = set()
            for p in array[2:]:
                index[a1][a2].add(p)

    return index


def read_mpindex_dblp(path="./data/dblp/"):
    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

    # print("{}{}.txt".format(path, PA_file))
    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1
    PT[:, 0] -= 1
    PT[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA_s = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                         shape=(paper_max, author_max),
                         dtype=np.float32)
    PT_s = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                         shape=(paper_max, term_max),
                         dtype=np.float32)

    transformer = TfidfTransformer()
    features = PA_s.transpose() * PT_s  # AT
    features = transformer.fit_transform(features)
    features = np.array(features.todense())

    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    labels_raw[:, 0] -= 1
    labels_raw[:, 1] -= 1
    labels = np.zeros(author_max)
    labels[labels_raw[:, 0]] = labels_raw[:, 1]

    reordered = np.random.permutation(labels_raw[:, 0])
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * 0.4))]
    idx_val = reordered[range(int(total_labeled * 0.4), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    folds = {'train':idx_train,'val':idx_val,'test':idx_test}

    return features, labels, folds


def load_edge_emb(path, schemes, n_dim=16, n_author=20000):
    data = np.load("{}edge{}.npz".format(path, n_dim))
    index = {}
    emb = {}
    for scheme in schemes:
        # print('number of authors: {}'.format(n_author))
        ind = sp.coo_matrix((np.arange(1,data[scheme].shape[0]+1),
                             (data[scheme][:, 0], data[scheme][:, 1])),
                            shape=(n_author, n_author),
                            dtype=np.long)
        ind = ind + ind.transpose()
        # print('ind generated')
        ind = torch.from_numpy(ind.todense())
        # print('ind generated')
        embedding = np.zeros(n_dim, dtype=np.float32)
        embedding = np.vstack((embedding, data[scheme][:, 2:]))
        emb[scheme] = torch.from_numpy(embedding).float()

        index[scheme] = ind
        print('loading edge embedding for {} complete'.format(scheme))

    return index, emb