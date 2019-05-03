import numpy as np
import scipy.sparse as sp
import torch
from utilities import *
from sklearn.feature_extraction.text import TfidfTransformer
import sys

def read_metapath_dblp(path="./data/dblp/"):
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


def read_embed(path="/home/danhao/Git/gcn/HINGCN/trunk/data/dblp/",
               emd_file="APC"):
    with open("{}{}.emd".format(path, emd_file)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    embedding = np.loadtxt("{}{}.emd".format(path, emd_file),
                           dtype=np.float32, skiprows=1)
    emd_index = {}
    for i in range(n_nodes):
        emd_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emd_index[i], 1:] for i in range(n_nodes)])
    features = torch.from_numpy(features)

    assert features.shape[1] == n_feature
    assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature


def read_mpindex_dblp(path="/home/danhao/Git/gcn/HINGCN/trunk/data/dblp/"):
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
    features = torch.FloatTensor(np.array(features.todense()))

    # read path sim
    adjs = []
    # p_APA = sp.load_npz("{}{}.npz".format(path, APA_file))
    # p_APAPA = sp.load_npz("{}{}.npz".format(path, APAPA_file))
    # p_APCPA = sp.load_npz("{}{}.npz".format(path, APCPA_file))
    # adjs.append(sparse_mx_to_torch_sparse_tensor(p_APA))
    # adjs.append(sparse_mx_to_torch_sparse_tensor(p_APAPA))
    # adjs.append(sparse_mx_to_torch_sparse_tensor(p_APCPA))

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

    # Read node embeddings
    node_emb = {}
    # n_nodes = 28871
    node_emb['APA'], n_nodes, n_feature = read_embed(path, APA_file)
    # node_emb['APAPA'] = read_embed(path, APAPA_file)
    # node_emb['APCPA'] = read_embed(path, APCPA_file)

    # build index
    PA[:, 0] += author_max
    PC[:, 0] += author_max
    PC[:, 1] += author_max + paper_max

    AP = np.copy(PA[:, [1, 0]])
    CP = np.copy(PC[:, [1, 0]])

    AP = AP[AP[:, 0].argsort()]
    CP = CP[CP[:, 0].argsort()]
    PA = PA[PA[:, 0].argsort()]
    PC = PC[PC[:, 0].argsort()]

    APi = []
    for i in range(n_nodes):
        arg = np.where(AP[:, 0] == i)[0]
        if len(arg):
            APi.append([arg[0], arg[-1] + 1])
        else:
            APi.append([0, 0])
    APi = np.asarray(APi)

    CPi = []
    for i in range(n_nodes):
        arg = np.where(CP[:, 0] == i)[0]
        if len(arg):
            CPi.append([arg[0], arg[-1] + 1])
        else:
            CPi.append([0, 0])

    CPi = np.asarray(CPi)

    PAi = []
    for i in range(n_nodes):
        arg = np.where(PA[:, 0] == i)[0]
        if len(arg):
            PAi.append([arg[0], arg[-1] + 1])
        else:
            PAi.append([0, 0])
    PAi = np.asarray(PAi)

    PCi = []
    for i in range(n_nodes):
        arg = np.where(PC[:, 0] == i)[0]
        if len(arg):
            PCi.append([arg[0], arg[-1] + 1])
        else:
            PCi.append([0, 0])
    PCi = np.asarray(PCi)

    index = {}
    index['AP'] = torch.LongTensor(AP)
    index['CP'] = torch.LongTensor(CP)
    index['PA'] = torch.LongTensor(PA)
    index['PC'] = torch.LongTensor(PC)
    index['APi'] = torch.LongTensor(APi)
    index['CPi'] = torch.LongTensor(CPi)
    index['PAi'] = torch.LongTensor(PAi)
    index['PCi'] = torch.LongTensor(PCi)

    return adjs, features, labels, idx_train, idx_val, idx_test, node_emb, index



# TODO: optimize path query; modify to torch tensors
def query_path(v, scheme, index, node_emb, sample_size=128):
    '''
    Generate metapaths with starting vertex v and path scheme scheme.

    :param v: query node
    :param scheme: 'APAPA'
    :param index: index['AP'] stores edges (a,p);
        index['APi'] stores offset for A.
    :param node_emb: node embeddings
    :param sample_size: number of neighbors sampled
    :return: paths: meta paths

    '''
    mp_len = len(scheme)

    paths = []
    tmp_paths = torch.LongTensor(np.asarray([[v]])).cuda()

    # find out index to be used:
    for n in range(mp_len - 1):
        # swap
        paths = tmp_paths

        # scheme[n]+scheme[n+1] is the corresponding index
        to_join = index[scheme[n] + scheme[n + 1]]
        to_join_ind = index[scheme[n] + scheme[n + 1] + 'i']

        tmp_paths = []
        # print(paths)
        for p in paths:
            last = p[-1]
            count = to_join_ind[last,1] - to_join_ind[last,0]
            # print(to_join_ind[last])
            rang = torch.arange(to_join_ind[last,0].item(),
                               to_join_ind[last,1].item(), dtype=torch.long).cuda()
            tmp_paths.append(torch.cat(
                (p.repeat(1, count).view(count, -1),
                 to_join[rang, 1].view(-1, 1)), dim=1))

        tmp_paths = torch.cat(tmp_paths)

    paths = tmp_paths

    # sort the paths according to the last vertex in path:
    paths = paths[paths[:, -1].argsort()]

    # neigh,cnt = np.unique(paths[:,-1],return_counts=True)
    x = paths[:, -1]
    x_unique = x.unique(sorted=True)
    x_unique_count = torch.stack([(x == x_u).sum() for x_u in x_unique])
    neigh = x_unique
    cnt = x_unique_count

    # groups = np.split(paths[:, 1:-1], np.cumsum(cnt)[:-1])
    groups = torch.split(paths[:, 1:-1], cnt.tolist())

    sampled = torch.LongTensor(np.random.choice
                               (neigh.shape[0], sample_size)).cuda()

    emb = [torch.sum(node_emb[scheme][groups[g].view(-1, 1)], dim=0) for g in sampled]
    emb = torch.cat(emb)

    return neigh[sampled], emb

#
# adjs, features, labels, idx_train, idx_val, idx_test, node_emb, index \
#     = read_mpindex_dblp(path="/home/danhao/Git/gcn/HINGCN/trunk/data/dblp/")
