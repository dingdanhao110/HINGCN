from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.feature_extraction.text import TfidfTransformer
from utilities import *
from hinmodel import *
from metapath import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=True,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--n_meta', type=int, default=3,
                    help='Number of meta-paths.')
parser.add_argument('--dim_mp', type=int, default=16,
                    help='Number of hidden units in layer2.')
parser.add_argument('--n_sample', type=int, default=16,
                    help='Dataset')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='alpha for leaky relu.')
parser.add_argument('--dataset', type=str, default='homograph',
                    help='Dataset')
parser.add_argument('--dataset_path', type=str, default='./data/dblp/',
                    help='Dataset')
parser.add_argument('--embedding_file', type=str, default='APA',
                    help='Dataset')
parser.add_argument('--label_file', type=str, default='author_label',
                    help='Dataset')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def read_embed(path="./data/dblp/",
               emd_file="APA"):
    with open("{}{}.emd".format(path, emd_file)) as f:
        n_nodes,n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes,n_feature))

    embedding = np.loadtxt("{}{}.emd".format(path, emd_file),
                              dtype=np.int32,skiprows=1)
    emd_index = {}
    for i in range(n_nodes):
        emd_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emd_index[i], 1:] for i in range(n_nodes)])

    assert features.shape[1] == n_feature
    assert features.shape[0] == n_nodes

    PA_file = "PA"
    PT_file = "PT"
    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PT[:, 0] -= 1
    PT[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.float32)
    PT = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                       shape=(paper_max, term_max),
                       dtype=np.float32)

    transformer = TfidfTransformer()
    AT = PA.transpose() * PT  # AT
    AT = transformer.fit_transform(AT)

    AT = AT.todense()
    AT = np.pad(AT, ((0,features.shape[0]-AT.shape[0]),(0,0)),'constant',constant_values=0 )

    print("number of nodes:{}, feature size:{}".format(AT.shape[0], AT.shape[1]))
    assert AT.shape[0] == n_nodes

    # features = AT
    features=np.concatenate((features,AT),axis=1)
    n_feature = features.shape[1]

    return n_nodes, n_feature, features


def read_graph(path="./data/dblp/", dataset="homograph", label_file="author_label", emb_file="APA"):
    print('Loading {} dataset...'.format(dataset))

    n_nodes, n_feature, features = read_embed(path,emb_file)
    features = torch.FloatTensor(features)

    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),dtype=np.int32)
    labels_raw[:, 0] -= 1
    labels_raw[:, 1] -= 1
    labels = np.zeros(n_nodes)
    labels[labels_raw[:, 0]] = labels_raw[:, 1]
    labels = torch.LongTensor(labels)

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.genfromtxt("{}{}.txt".format(path, dataset),
                                    dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                       shape=(n_nodes, n_nodes),
                       dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    reordered = np.random.permutation(labels_raw[:, 0])
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * 0.4))]
    idx_val = reordered[range(int(total_labeled * 0.4), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

def read_graph2(path="./data/dblp/", dataset="homograph", label_file="author_label", emb_file="APA"):
    print('Loading {} dataset...'.format(dataset))

    n_nodes, n_feature, features = read_embed(path,emb_file)
    features = torch.FloatTensor(features)

    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),dtype=np.int32)
    labels_raw[:, 0] -= 1
    labels_raw[:, 1] -= 1
    labels = np.zeros(n_nodes)
    labels[labels_raw[:, 0]] = labels_raw[:, 1]
    labels = torch.LongTensor(labels)

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.genfromtxt("{}{}.txt".format(path, dataset),
                                    dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                       shape=(n_nodes, n_nodes),
                       dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    reordered = np.random.permutation(labels_raw[:, 0])
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * 0.4))]
    idx_val = reordered[range(int(total_labeled * 0.4), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def read_graph_yelp(path="./data/yelp/", dataset="homograph",
                    label_file="true_cluster", emb_file="RBUK_16"):
    print('Loading {} dataset...'.format(dataset))

    with open("{}{}.emb".format(path, emb_file)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    embedding = np.loadtxt("{}{}.emb".format(path, emb_file),
                           dtype=np.int32, skiprows=1)
    emd_index = {}
    for i in range(n_nodes):
        emd_index[embedding[i, 0]] = i

    embedding = np.asarray([embedding[emd_index[i], 1:] for i in range(n_nodes)])

    assert embedding.shape[1] == n_feature
    assert embedding.shape[0] == n_nodes
    embedding = torch.FloatTensor(embedding)

    features = np.genfromtxt("{}{}.txt".format(path, 'attributes'),
                                    dtype=np.float)
    features = np.pad(features, ((0, embedding.shape[0] - features.shape[0]), (0, 0)), 'constant', constant_values=0)

    features = torch.FloatTensor(features[:,:2])
    features = torch.cat([features,embedding], dim=1)

    # build graph
    # idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.genfromtxt("{}{}.txt".format(path, dataset),
                                    dtype=np.int32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                       shape=(n_nodes, n_nodes),
                       dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    labels = np.genfromtxt("{}{}.txt".format(path, label_file),
                           dtype=np.int32)
    reordered = np.random.permutation(np.arange(labels.shape[0]))
    total_labeled = labels.shape[0]

    idx_train = reordered[range(int(total_labeled * 0.4))]
    idx_val = reordered[range(int(total_labeled * 0.4), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    labels = torch.LongTensor(labels)

    return adj, features, labels, idx_train, idx_val, idx_test

# Load data
adj, features, labels, idx_train, idx_val, idx_test = \
    read_graph_yelp()

print('Read data finished!')

# Model and optimizer
model = GCN(n_nodes=features.shape[0],
            nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout,
            prep=True,
            emb_dim=128,
            )
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

print('Model init finished!')

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
    labels = labels.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features,adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features,adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
