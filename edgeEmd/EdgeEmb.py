'''
Indexing: index[a1,a2,0] ~ index[a1,a2,1] is the range of embeddings of multi-edges between a1 and a2

get matrix of embeddings: emb[ np.arange(index[a1,a2,0] , index[a1,a2,1]) ]

'''
import sys
sys.path.append('../')
from metapath import *


def dump_edge_emb(path='../data/dblp/'):
    # dump APA
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    adjs, features, labels, idx_train, idx_val, idx_test, node_emb, index \
        = read_mpindex_dblp(path=path)

    n_author = features.shape[0]
    emb_len = node_emb['APA'].shape[1]

    # APA
    # with open("{}{}edge{}.emb".format(path, APA_file,emb_len), mode='w') as f:
    scheme = 'APA'
    APA = index['APA']

    APA_emb = []
    for a1 in APA.keys():
        for a2 in APA[a1]:
            tmp = [node_emb[scheme][p].numpy() for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0)
            if a1 < a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp)))
    APA_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(APA_file))

    # APAPA
    APAPA_emb = []
    # with open("{}{}edge{}.emb".format(path, APAPA_file,emb_len), mode='w') as f:
    scheme = 'APAPA'
    scheme1 = scheme[0:3]
    scheme2 = scheme[2:5]
    ind1 = index[scheme1]
    ind2 = index[scheme2]

    for v in range(n_author):
        result = {}
        if v not in ind1.keys():
            # print (v)
            continue
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

        for a in result:
            if v < a:
                APAPA_emb.append(np.concatenate(([v, a], result[a].numpy())))
            # f.write('{} {} '.format(v, a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APAPA_emb = np.asarray(APAPA_emb)
    print("compute edge embeddings {} complete".format(APAPA_file))

    # with open("{}{}edge{}.emb".format(path, APCPA_file,emb_len), mode='w') as f:
    scheme = 'APCPA'
    scheme1 = scheme[0:3]
    scheme2 = scheme[2:5]
    ind1 = index[scheme1]
    ind2 = index[scheme2]
    APCPA_emb = []
    for v in range(n_author):
        result = {}
        if v not in ind1.keys():
            # print (v)
            continue
        if len(ind1[v]) == 0:
            continue
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

        for a in result:
            if v < a:
                APCPA_emb.append(np.concatenate(([v, a], result[a].numpy())))
            # f.write('{} {} '.format(v,a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APCPA_emb = np.asarray(APCPA_emb)
    print("compute edge embeddings {} complete".format(APCPA_file))

    np.savez("{}edge{}.npz".format(path, emb_len),
             APA=APA_emb, APAPA=APAPA_emb, APCPA=APCPA_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass


def load_edge_emb(path="../data/dblp/", schemes=['APA', 'APAPA', 'APCPA'], n_dim=16,n_author=20000):
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


dump_edge_emb()
# index, emb = load_edge_emb()
