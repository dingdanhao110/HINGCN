from sklearn import metrics
import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def read_metapath(path="../data/cora/", dataset="cora", num_mps=1):
    """read metapath file, A1~A2 pairs"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(np.where(labels)[1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    adjs = []
    for path_idx in range(num_mps):
        edges_unordered = np.genfromtxt("{}{}_{}.metapaths".format(path, dataset, path_idx),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(features.shape[0], features.shape[0]),
                            dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        adjs.append(adj.unsqueeze(0))
    adjs = torch.cat(adjs)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adjs, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels).cpu().detach()
    return metrics.f1_score(labels.cpu().detach(), preds, average="macro")


def read_metapath_raw(path="../data/cora/", dataset="cora", num_mps=1):
    """read metapath file, A1,A2,pathsim triples, return adj are not normalized"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense()))

    labels = encode_onehot(idx_features_labels[:, -1])
    labels = torch.LongTensor(np.where(labels)[1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    adjs = []
    for path_idx in range(num_mps):
        edges_unordered = np.genfromtxt("{}{}_{}.metapaths".format(path, dataset, path_idx),
                                        dtype=np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                         dtype=np.int32).reshape(edges_unordered.shape)

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(features.shape[0], features.shape[0]),
                            dtype=np.bool)

        # build symmetric adjacency matrix
        adj = adj + adj.T
        # adj = (adj + sp.eye(adj.shape[0])) # no normalization
        adj = sparse_mx_to_torch_sparse_tensor(adj)

        adjs.append(adj)
        # adjs.append(adj.unsqueeze(0))
    # adjs = torch.cat(adjs)

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adjs, features, labels, idx_train, idx_val, idx_test


def pathsim(A):
    value = []
    x, y = A.nonzero()
    for i, j in zip(x, y):
        value.append(2 * A[i, j] / (A[i, i] + A[j, j]))
    return sp.coo_matrix((value, (x, y)))
