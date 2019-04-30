from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

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
parser.add_argument('--epochs', type=int, default=20,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.1,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--n_meta', type=int, default=1,
                    help='Number of meta-paths.')
parser.add_argument('--dim_mp', type=int, default=16,
                    help='Number of hidden units in layer2.')
parser.add_argument('--n_sample', type=int, default=16,
                    help='Dataset')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.8,
                    help='alpha for leaky relu.')
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset')
parser.add_argument('--dataset_path', type=str, default='../pygcn/data/cora/',
                    help='Dataset')
parser.add_argument('--model', type=str, default='hingcn',
                    help='Model used in training')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adjs, features, labels, idx_train, idx_val, idx_test, node_emb, index \
    = read_mpindex_dblp(path="/home/danhao/Git/gcn/HINGCN/trunk/data/dblp/")

print('Read data finished!')

# Model and optimizer
model = HINGCN_edge(nfeat=features.shape[1],
            nhid=args.hidden,
            nmeta=args.n_meta,
            dim_mp=args.dim_mp,
            edge_dim=node_emb['APA'].shape[1],
            schemes=['APA'],
            nclass=labels.max().item() + 1,
            alpha=args.alpha,
            dropout=args.dropout,
            adjs=[],
            bias=True,
            concat=True,
            samples=args.n_sample
                  )
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

print('Model init finished!')

if args.cuda:
    model.cuda()
    features = features.cuda()
    adjs = [adj.cuda() for adj in adjs]
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, index, node_emb, n_sample=args.n_sample)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, index, node_emb, n_sample=args.n_sample)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, index, node_emb, n_sample=args.n_sample)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# def mini_batch(n_epochs = 100, batch_size = 128):
#
#     for epoch in range(n_epochs):
#         t = time.time()
#         model.train()
#         # X is a torch Variable
#         permutation = torch.randperm(idx_train.shape[0])
#
#         for i in range(0, idx_train.size()[0], batch_size):
#             optimizer.zero_grad()
#
#             indices = permutation[i:i + batch_size]
#             batch_x, batch_y = idx_train[indices], labels[indices]
#
#             # in case you wanted a semi-full example
#             outputs = model.forward(features, adjs, batch_x)
#             loss = F.nll_loss(outputs, batch_y)
#             # acc_train = accuracy(outputs, batch_y)
#
#             loss.backward()
#             optimizer.step()
#         model.eval()
#         output = model(features, adjs, idx_val)
#         loss_val = F.nll_loss(output, labels[idx_val])
#         acc_val = accuracy(output, labels[idx_val])
#         print('Epoch: {:04d}'.format(epoch + 1),
#               # 'loss_train: {:.4f}'.format(loss_train.item()),
#               # 'acc_train: {:.4f}'.format(acc_train.item()),
#               'loss_val: {:.4f}'.format(loss_val.item()),
#               'acc_val: {:.4f}'.format(acc_val.item()),
#               'time: {:.4f}s'.format(time.time() - t))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()