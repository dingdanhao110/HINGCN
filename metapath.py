import numpy as np
import scipy.sparse as sp
import torch
from utilities import *
from sklearn.feature_extraction.text import TfidfTransformer


def read_metapath_dblp(path="/home/danhao/Git/gcn/HINGCN/trunk/data/dblp/"):
    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

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
    conf_max = max(PC[:, 1]) + 1
    term_max = max(PT[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.float32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.float32)
    PT = sp.coo_matrix((np.ones(PT.shape[0]), (PT[:, 0], PT[:, 1])),
                       shape=(paper_max, term_max),
                       dtype=np.float32)
    # PA = sparse_mx_to_torch_sparse_tensor(PA)
    # PC = sparse_mx_to_torch_sparse_tensor(PC)
    # PT = sparse_mx_to_torch_sparse_tensor(PT)

    transformer = TfidfTransformer()
    features = PA.transpose() * PT  # AT
    features = transformer.fit_transform(features)
    features = torch.FloatTensor(np.array(features.todense()))

    # read path sim
    adjs = []
    p_APA = sp.load_npz("{}{}.npz".format(path, APA_file))
    p_APAPA = sp.load_npz("{}{}.npz".format(path, APAPA_file))
    p_APCPA = sp.load_npz("{}{}.npz".format(path, APCPA_file))
    adjs.append(sparse_mx_to_torch_sparse_tensor(p_APA))
    adjs.append(sparse_mx_to_torch_sparse_tensor(p_APAPA))
    adjs.append(sparse_mx_to_torch_sparse_tensor(p_APCPA))


    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    labels_raw[:, 0] -= 1
    labels_raw[:, 1] -= 1
    labels = np.zeros(author_max)
    labels[labels_raw[:, 0]] = labels_raw[:, 1]
    labels = torch.LongTensor(labels)

    reordered = np.random.permutation(labels_raw[:, 0])
    total_labeled = labels_raw.shape[0]

    idx_train = reordered[range(int(total_labeled * 0.4))]
    idx_val = reordered[range(int(total_labeled * 0.4), int(total_labeled * 0.8))]
    idx_test = reordered[range(int(total_labeled * 0.8), total_labeled)]

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adjs, features, labels, idx_train, idx_val, idx_test
