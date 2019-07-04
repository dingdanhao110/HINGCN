import numpy as np


def read_embed(path="./data/dblp/",
               emb_file="RUBK"):
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

def gen_homograph(path = "../../../data/yelp/", out_file = "homograph"):

    label_file = "attributes"
    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                   dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                   dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                   dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    rate_max = max(RB[:, 0]) + 1   #33360
    busi_max = max(RB[:, 1]) + 1   #2614
    key_max = max(RK[:, 1]) + 1    #82
    user_max = max(RU[:, 1]) + 1   #1286

    # busi: [0,busi_max)
    # rate: [busi_max,busi_max+rate_max)
    # key: [busi_max+rate_max,busi_max+rate_max+key_max)
    # user: [busi_max+rate_max+key_max,busi_max+rate_max+key_max+user_max)

    RU[:, 0] += busi_max
    RB[:, 0] += busi_max
    RK[:, 0] += busi_max

    RK[:, 1] += busi_max+rate_max

    RU[:, 1] += busi_max+rate_max+key_max

    edges = np.concatenate((RB,RK,RU),axis=0)

    np.savetxt("{}{}.txt".format(path, out_file),edges,fmt='%u')



def dump_yelp_edge_emb(path='../../../data/yelp/'):
    # dump APA
    label_file = "attributes"
    RB_file = "RB"
    RK_file = "RK"
    RU_file = "RU"

    RB = np.genfromtxt("{}{}.txt".format(path, RB_file),
                       dtype=np.int32)
    RK = np.genfromtxt("{}{}.txt".format(path, RK_file),
                       dtype=np.int32)
    RU = np.genfromtxt("{}{}.txt".format(path, RU_file),
                       dtype=np.int32)
    RB[:, 0] -= 1
    RB[:, 1] -= 1
    RK[:, 0] -= 1
    RK[:, 1] -= 1
    RU[:, 0] -= 1
    RU[:, 1] -= 1

    # BR = np.copy(RB[:, [1, 0]])
    # KR = np.copy(RK[:, [1, 0]])
    # UR = np.copy(RU[:, [1, 0]])
    #
    # BR = BR[BR[:, 0].argsort()]
    # KR = KR[KR[:, 0].argsort()]
    # UR = UR[UR[:, 0].argsort()]

    #--
    #build index for 2hop adjs

    RBi={}
    BRi={}
    RKi={}
    KRi={}
    RUi={}
    URi={}

    for i in range(RB.shape[0]):
        r=RB[i,0]
        b=RB[i,1]

        if r not in RBi:
            RBi[r]=set()
        if b not in BRi:
            BRi[b]=set()

        RBi[r].add(b)
        BRi[b].add(r)

    for i in range(RK.shape[0]):
        r=RK[i,0]
        k=RK[i,1]

        if r not in RKi:
            RKi[r]=set()
        if k not in KRi:
            KRi[k]=set()

        RKi[r].add(k)
        KRi[k].add(r)

    for i in range(RU.shape[0]):
        r=RU[i,0]
        u=RU[i,1]

        if r not in RUi:
            RUi[r]=set()
        if u not in URi:
            URi[u]=set()

        RUi[r].add(u)
        URi[u].add(r)

    BRUi={}
    URBi={}

    BRKi={}
    KRBi={}

    for b in BRi:
        for r in BRi[b]:
            if r not in RUi:
                continue
            for u in RUi[r]:
                if b not in BRUi:
                    BRUi[b] ={}
                if u not in URBi:
                    URBi[u] ={}

                if u not in BRUi[b]:
                    BRUi[b][u]=set()
                if b not in URBi[u]:
                    URBi[u][b]=set()

                BRUi[b][u].add(r)
                URBi[u][b].add(r)

    for b in BRi:
        for r in BRi[b]:
            if r not in RKi:
                continue
            for k in RKi[r]:
                if b not in BRKi:
                    BRKi[b]={}
                if k not in KRBi:
                    KRBi[k]={}
                if k not in BRKi[b]:
                    BRKi[b][k]=set()
                if b not in KRBi[k]:
                    KRBi[k][b]=set()
                BRKi[b][k].add(r)
                KRBi[k][b].add(r)


    rate_max = max(RB[:, 0]) + 1  # 33360
    busi_max = max(RB[:, 1]) + 1  # 2614
    key_max = max(RK[:, 1]) + 1  # 82
    user_max = max(RU[:, 1]) + 1  # 1286

    n_busi = busi_max
    node_emb, n_nodes, emb_len = read_embed(path=path,emb_file="RBUK_16")

    # brurb;
    BRURB_emb = []
    for v in range(n_busi):
        result = {}
        count = {}
        if v not in BRUi.keys():
            # print (v)
            continue
        for u in BRUi[v]:
            np1 = len(BRUi[v][u])
            edge1 = [node_emb[p] for p in BRUi[v][u]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for b in URBi[u].keys():
                np2 = len(URBi[u][b])
                edge2 = [node_emb[p] for p in URBi[u][b]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if b not in result:
                    result[b] = node_emb[u] * (np2 * np1)
                else:
                    result[b] += node_emb[u] * (np2 * np1)
                result[b] += edge1 * np2
                result[b] += edge2 * np1
                if b not in count:
                    count[b]=0
                count[b] += np1*np2

        for b in result:
            if v < b:
                BRURB_emb.append(np.concatenate(([v, b], result[b], [count[b]])))
    BRURB_emb = np.asarray(BRURB_emb)
    m = np.max(BRURB_emb[:, -1])
    BRURB_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format('BRURB'))

    #  brkrb
    BRKRB_emb = []

    for v in range(n_busi):
        result = {}
        count = {}
        if v not in BRKi.keys():
            # print (v)
            continue
        for k in BRKi[v].keys():
            np1 = len(BRKi[v][k])
            edge1 = [node_emb[p] for p in BRKi[v][k]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            for b in KRBi[k].keys():
                np2 = len(KRBi[k][b])
                edge2 = [node_emb[p] for p in KRBi[k][b]]
                edge2 = np.sum(np.vstack(edge2), axis=0)  # edge2: the emd between a1 and a2
                if b not in result:
                    result[b] = node_emb[k] * (np2 * np1)
                else:
                    result[b] += node_emb[k] * (np2 * np1)
                if b not in count:
                    count[b]=0
                result[b] += edge1 * np2
                result[b] += edge2 * np1
                count[b] += np1*np2
        for b in result:
            if v < b:
                BRKRB_emb.append(np.concatenate(([v, b], result[b], [count[b]] )))
    BRKRB_emb = np.asarray(BRKRB_emb)
    m = np.max(BRKRB_emb[:, -1])
    BRKRB_emb[:, -1] /= m
    print("compute edge embeddings {} complete".format('BRKRB'))
    emb_len = BRKRB_emb.shape[1] - 2
    np.savez("{}edge{}.npz".format(path, emb_len),
             BRURB=BRURB_emb, BRKRB=BRKRB_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass


# gen_homograph()
dump_yelp_edge_emb(path='data/yelp/')