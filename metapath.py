import numpy as np
import scipy.sparse as sp
import torch
from utilities import *
import random
from sklearn.feature_extraction.text import TfidfTransformer


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
    features = torch.from_numpy(features)

    assert features.shape[1] == n_feature
    assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature


def dump_2hop_index(index, path="./data/dblp/", file="APA"):
    with open("{}{}.ind".format(path, file), mode='w') as f:

        for a1 in index:
            for a2 in index[a1]:
                f.write('{} {}'.format(a1, a2))
                for p in index[a1][a2]:
                    f.write(' {}'.format(p))
                f.write('\n')

    print("dump index {} complete".format(file))
    pass


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


def gen_2hop_index(path="./data/dblp/"):
    PA_file = "PA"
    PC_file = "PC"
    # PT_file = "PT"
    APA_file = "APA"
    APC_file = "APC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    # PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
    #                    dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1
    # PT[:, 0] -= 1
    # PT[:, 1] -= 1

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1
    # term_max = max(PT[:, 1]) + 1

    n_nodes = paper_max + author_max + conf_max
    assert n_nodes == 28871

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

    APA_index = {}

    for a in range(author_max):
        APA_index[a] = {}
        rang = np.arange(APi[a, 0],
                         APi[a, 1])

        for p in AP[rang, 1]:
            a2i = np.arange(PAi[p, 0], PAi[p, 1])
            for a2 in PA[a2i, 1]:
                if a == a2:
                    continue
                if a2 not in APA_index[a]:
                    APA_index[a][a2] = set()
                APA_index[a][a2].add(p)

    APC_index = {}

    for a in range(author_max):
        APC_index[a] = {}
        rang = np.arange(APi[a, 0],
                         APi[a, 1])

        for p in AP[rang, 1]:
            ci = np.arange(PCi[p, 0], PCi[p, 1])
            for c in PC[ci, 1]:
                if c not in APC_index[a]:
                    APC_index[a][c] = set()
                APC_index[a][c].add(p)

    CPA_index = {}

    for c in range(author_max+paper_max,author_max+paper_max+conf_max):

        CPA_index[c] = {}
        rang = np.arange(CPi[c, 0],
                         CPi[c, 1])

        for p in CP[rang, 1]:
            ai = np.arange(PAi[p, 0], PAi[p, 1])
            for a in PA[ai, 1]:
                if a not in CPA_index[c]:
                    CPA_index[c][a] = set()
                CPA_index[c][a].add(p)

    print("gen index complete")

    # dump_2hop_index(APA_index, file="APA")
    # dump_2hop_index(APC_index, file="APC")
    # dump_2hop_index(CPA_index, file="CPA")

    # print(APA_index.__sizeof__())
    # print(APC_index.__sizeof__())
    # print(CPA_index.__sizeof__())

    return APA_index, APC_index, CPA_index


def read_mpindex_dblp(path="./data/dblp/"):
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

    # read path count
    adjs = {}
    # adjs['APA'] = sp.load_npz("{}{}_cnt.npz".format(path, APA_file))
    # adjs['APAPA'] = sp.load_npz("{}{}_cnt.npz".format(path, APAPA_file))
    # adjs['APCPA'] = sp.load_npz("{}{}_cnt.npz".format(path, APCPA_file))

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
    node_emb['APA'], n_nodes, n_feature = read_embed(path, 'APC_16')
    node_emb['APAPA'] = node_emb['APA']
    node_emb['APCPA'] = node_emb['APA']
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

    APA_index = load_2hop_index(path=path,file="APA")
    APC_index = load_2hop_index(path=path,file='APC')
    CPA_index = load_2hop_index(path=path,file='CPA')

    index = {}
    index['AP'] = torch.LongTensor(AP)
    index['CP'] = torch.LongTensor(CP)
    index['PA'] = torch.LongTensor(PA)
    index['PC'] = torch.LongTensor(PC)
    index['APi'] = torch.LongTensor(APi)
    index['CPi'] = torch.LongTensor(CPi)
    index['PAi'] = torch.LongTensor(PAi)
    index['PCi'] = torch.LongTensor(PCi)
    index['APA'] = APA_index
    index['APC'] = APC_index
    index['CPA'] = CPA_index
    # index['adjs'] = adjs

    return [], features, labels, idx_train, idx_val, idx_test, node_emb, index


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
            count = to_join_ind[last, 1] - to_join_ind[last, 0]
            # print(to_join_ind[last])
            rang = torch.arange(to_join_ind[last, 0].item(),
                                to_join_ind[last, 1].item(), dtype=torch.long).cuda()
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


def query_path_indexed(v, scheme, index, node_emb, sample_size=128):
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
    # emb_len = node_emb['APA'].shape[1]
    if mp_len == 3:
        # find out index to be used:
        # cnt = index['adjs'][scheme]
        ind = index[scheme]

        neigh = []
        result = []

        if sample_size > len(ind[v]):
            sample_size = len(ind[v])

        for a in random.sample(ind[v].keys(), sample_size):
            neigh.append(a)
            tmp = []
            for p in ind[v][a]:
                tmp.append(node_emb[scheme][p])
            tmp = torch.sum(torch.stack(tmp), dim=0)
            result.append(tmp)
        result = torch.stack(result)
        neigh = torch.LongTensor(neigh)
        assert neigh.shape[0] == result.shape[0]

        return neigh, result
    else:
        # len==5
        # example metapath instance: v-p-a1-p-a2
        # find out index to be used:
        scheme1 = scheme[0:3]
        scheme2 = scheme[2:5]
        ind1 = index[scheme1]
        ind2 = index[scheme2]

        result = {}
        for a1 in ind1[v].keys():
            np1 = len(ind1[v][a1])
            edge1 = [node_emb[scheme][p] for p in ind1[v][a1]]
            edge1 = torch.sum(torch.stack(edge1), dim=0)  # edge1: the emd between v and a1

            for a2 in ind2[a1].keys():
                np2 = len(ind2[a1][a2])
                edge2 = [node_emb[scheme][p] for p in ind2[a1][a2]]
                edge2 = torch.sum(torch.stack(edge2), dim=0)  # edge2: the emd between a1 and a2
                if a2 not in result:
                    result[a2] = node_emb[scheme][a1] * (np2 * np1)
                else:
                    result[a2] += node_emb[scheme][a1] * (np2 * np1)
                result[a2] += edge1 * np2
                result[a2] += edge2 * np1

        res = []
        neigh = []
        for nei,emb in result.items():
            neigh.append(nei)
            res.append(emb)
        neigh = torch.LongTensor(neigh)
        res = torch.stack(res)
        return neigh,res

#
# adjs, features, labels, idx_train, idx_val, idx_test, node_emb, index \
#     = read_mpindex_dblp(path="./data/dblp/")

# APA_index, APC_index, CPA_index = gen_2hop_index()
