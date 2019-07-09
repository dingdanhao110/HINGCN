import numpy as np
import scipy.sparse as sp
import torch
import random
from sklearn.feature_extraction.text import TfidfTransformer


def clean_dblp(path='./data/dblp/',new_path='./data/dblp2/'):

    label_file = "author_label"
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PT = np.genfromtxt("{}{}.txt".format(path, PT_file),
                       dtype=np.int32)
    
    labels_raw = np.genfromtxt("{}{}.txt".format(path, label_file),
                               dtype=np.int32)
    
    A = {}
    for i,a in enumerate(labels_raw[:,0]):
        A[a]=i+1
    print(len(A))
    PA_new = np.asarray([[PA[i,0],A[PA[i,1]]] for i in range(PA.shape[0]) if PA[i,1] in A])
    PC_new = PC
    PT_new = PT

    labels_new = np.asarray([[A[labels_raw[i,0]],labels_raw[i,1]] for i in range(labels_raw.shape[0]) if labels_raw[i,0] in A])

    np.savetxt("{}{}.txt".format(new_path, PA_file),PA_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, PC_file),PC_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, PT_file),PT_new,fmt='%i')
    np.savetxt("{}{}.txt".format(new_path, label_file),labels_new,fmt='%i')

def gen_homograph():
    path = "data/dblp2/"
    out_file = "homograph"

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

    PA[:, 0] += author_max
    PC[:, 0] += author_max
    PC[:, 1] += author_max+paper_max

    edges = np.concatenate((PA,PC),axis=0)

    np.savetxt("{}{}.txt".format(path, out_file),edges,fmt='%u')

def read_embed(path="data/dblp2/",
               emb_file="APC",emb_len=16):
    with open("{}{}_{}.emb".format(path, emb_file,emb_len)) as f:
        n_nodes, n_feature = map(int, f.readline().strip().split())
    print("number of nodes:{}, embedding size:{}".format(n_nodes, n_feature))

    embedding = np.loadtxt("{}{}_{}.emb".format(path, emb_file,emb_len),
                           dtype=np.float32, skiprows=1)
    emb_index = {}
    for i in range(n_nodes):
        emb_index[embedding[i, 0]] = i

    features = np.asarray([embedding[emb_index[i], 1:] for i in range(n_nodes)])

    assert features.shape[1] == n_feature
    assert features.shape[0] == n_nodes

    return features, n_nodes, n_feature

def dump_edge_emb(path='data/dblp2/'):
    # dump APA
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    node_emb,n_nodes,n_emb =read_embed()

    PA_file = "PA"
    PC_file = "PC"

    PA = np.genfromtxt("{}{}.txt".format(path, PA_file),
                       dtype=np.int32)
    PC = np.genfromtxt("{}{}.txt".format(path, PC_file),
                       dtype=np.int32)
    PA[:, 0] -= 1
    PA[:, 1] -= 1
    PC[:, 0] -= 1
    PC[:, 1] -= 1

    PAi={}
    APi={}
    PCi={}
    CPi={}

    for i in range(PA.shape[0]):
        p=PA[i,0]
        a=PA[i,1]

        if p not in PAi:
            PAi[p]=set()
        if a not in APi:
            APi[a]=set()

        PAi[p].add(a)
        APi[a].add(p)

    for i in range(PC.shape[0]):
        p=PC[i,0]
        c=PC[i,1]

        if p not in PCi:
            PCi[p]=set()
        if c not in CPi:
            CPi[c]=set()

        PCi[p].add(c)
        CPi[c].add(p)

    APAi={}
    APCi={}
    CPAi={}

    for v in APi:
        for p in APi[v]:
            if p not in PAi:
                continue
            for a in PAi[p]:
                if a not in APAi:
                    APAi[a] ={}
                if v not in APAi:
                    APAi[v] ={}

                if v not in APAi[a]:
                    APAi[a][v]=set()
                if a not in APAi[v]:
                    APAi[v][a]=set()

                APAi[a][v].add(p)
                APAi[v][a].add(p)
    
    for v in APi:
        for p in APi[v]:
            if p not in PCi:
                continue
            for c in PCi[p]:
                if v not in APCi:
                    APCi[v] ={}
                if c not in CPAi:
                    CPAi[c] ={}

                if c not in APCi[v]:
                    APCi[v][c]=set()
                if v not in CPAi[c]:
                    CPAi[c][v]=set()

                CPAi[c][v].add(p)
                APCi[v][c].add(p)



    ## APAPA; vpa1pa2
    #APAPA_emb = []
    #for v in APAi:
    #    result = {}
    #    count = {}
    #    for a1 in APAi[v]:
    #        np1 = len(APAi[v][a1])
    #        edge1 = [node_emb[p] for p in APAi[v][a1]]
    #        edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

    #        for a2 in APAi[a1].keys():
    #            np2 = len(APAi[a1][a2])
    #            edge2 = [node_emb[p] for p in APAi[a1][a2]]
    #            edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
    #            if a2 not in result:
    #                result[a2] = node_emb[a2] * (np2 * np1)
    #            else:
    #                result[a2] += node_emb[a2] * (np2 * np1)
    #            result[a2] += edge1 * np2
    #            result[a2] += edge2 * np1
    #            if a2 not in count:
    #                count[a2]=0
    #            count[a2] += np1*np2

    #    for a2 in result:
    #        if v <= a2:
    #            APAPA_emb.append(np.concatenate(([v, a2], result[a2]/count[a2], [count[a2]])))
    #APAPA_emb = np.asarray(APAPA_emb)
    #m = np.max(APAPA_emb[:, -1])
    #APAPA_emb[:, -1] /= m
    #print("compute edge embeddings {} complete".format('APAPA'))    

    # APA
    APA = APAi

    APA_emb = []
    for a1 in APA.keys():
        for a2 in APA[a1]:
            tmp = [node_emb[p] for p in APA[a1][a2]]
            tmp = np.sum(tmp, axis=0)/len(APA[a1][a2])
            tmp += node_emb[a1]+node_emb[a2]
            tmp /= 3
            if a1 <= a2:
                APA_emb.append(np.concatenate(([a1, a2], tmp, [len(APA[a1][a2])])))
    APA_emb = np.asarray(APA_emb)
    print("compute edge embeddings {} complete".format(APA_file))

    # APAPA
    APAPA_emb = []
    ind1 = APAi
    ind2 = APAi

    for v in ind1:
        result = {}
        count = {}
        for a1 in ind1[v].keys():
            np1 = len(ind1[v][a1])
            edge1 = [node_emb[p] for p in ind1[v][a1]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for a2 in ind2[a1].keys():
                np2 = len(ind2[a1][a2])
                edge2 = [node_emb[p] for p in ind2[a1][a2]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if a2 not in result:
                    result[a2] = node_emb[a1] * (np2 * np1)
                else:
                    result[a2] += node_emb[a1] * (np2 * np1)
                result[a2] += edge1 * np2
                result[a2] += edge2 * np1
                if a2 not in count:
                    count[a2]=0
                count[a2] += np1*np2

        for a in result:
            if v <= a:
                APAPA_emb.append(np.concatenate(([v, a], (result[a]/count[a]+node_emb[a]+node_emb[v])/5
                                                 ,[count[a]])))
            # f.write('{} {} '.format(v, a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APAPA_emb = np.asarray(APAPA_emb)
    m = np.max(APAPA_emb[:, -1])
    APAPA_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format(APAPA_file))

    #APCPA
    ind1 = APCi
    ind2 = CPAi
    APCPA_emb = []
    for v in ind1:
        result = {}
        count = {}
        if len(ind1[v]) == 0:
            continue
        for a1 in ind1[v].keys():
            np1 = len(ind1[v][a1])
            edge1 = [node_emb[p] for p in ind1[v][a1]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for a2 in ind2[a1].keys():
                np2 = len(ind2[a1][a2])
                edge2 = [node_emb[p] for p in ind2[a1][a2]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if a2 not in result:
                    result[a2] = node_emb[a1] * (np2 * np1)
                else:
                    result[a2] += node_emb[a1] * (np2 * np1)
                if a2 not in count:
                    count[a2]=0
                result[a2] += edge1 * np2
                result[a2] += edge2 * np1
                count[a2] += np1*np2

        
        for a in result:
            if v <= a:
                APCPA_emb.append(np.concatenate(([v, a], (result[a]/count[a]+node_emb[a]+node_emb[v])/5,
                                                 [count[a]])))
            # f.write('{} {} '.format(v,a))
            # f.write(" ".join(map(str, result[a].numpy())))
            # f.write('\n')
    APCPA_emb = np.asarray(APCPA_emb)
    m = np.max(APCPA_emb[:, -1])
    APCPA_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format(APCPA_file))
    emb_len=APA_emb.shape[1]-2
    np.savez("{}edge{}.npz".format(path, emb_len),
             APA=APA_emb, APAPA=APAPA_emb, APCPA=APCPA_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass

#clean_dblp()
#gen_homograph()
dump_edge_emb()