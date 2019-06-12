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


def gen_homograph(path="../../../data/yago/", out_file="homograph"):
    label_file = "labels"
    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)     #1465
    n_actor = len(actors)    #4019
    n_director = len(directors) #1093
    n_writer = len(writers)    #1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i+n_movie for (i, a) in enumerate(actors)}
    director_dict = {a: i+n_movie+n_actor for (i, a) in enumerate(directors)}
    writer_dict = {a: i+n_movie+n_actor+n_director for (i, a) in enumerate(writers)}


    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]] ])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    edges = np.concatenate((MA, MD, MW), axis=0)

    np.savetxt("{}{}.txt".format(path, out_file), edges, fmt='%u')


def dump_yago_edge_emb(path='../../../data/yago/'):
    # dump APA
    label_file = "labels"
    MA_file = "movie_actor"
    MD_file = "movie_director"
    MW_file = "movie_writer"

    movies = []
    actors = []
    directors = []
    writers = []

    with open('{}{}.txt'.format(path, "movies"), mode='r', encoding='UTF-8') as f:
        for line in f:
            movies.append(line.split()[0])

    with open('{}{}.txt'.format(path, "actors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            actors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "directors"), mode='r', encoding='UTF-8') as f:
        for line in f:
            directors.append(line.split()[0])

    with open('{}{}.txt'.format(path, "writers"), mode='r', encoding='UTF-8') as f:
        for line in f:
            writers.append(line.split()[0])

    n_movie = len(movies)  # 1465
    n_actor = len(actors)  # 4019
    n_director = len(directors)  # 1093
    n_writer = len(writers)  # 1458

    movie_dict = {a: i for (i, a) in enumerate(movies)}
    actor_dict = {a: i + n_movie for (i, a) in enumerate(actors)}
    director_dict = {a: i + n_movie + n_actor for (i, a) in enumerate(directors)}
    writer_dict = {a: i + n_movie + n_actor + n_director for (i, a) in enumerate(writers)}

    MA = []
    with open('{}{}.txt'.format(path, MA_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MA.append([movie_dict[arr[0]], actor_dict[arr[1]]])

    MD = []
    with open('{}{}.txt'.format(path, MD_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MD.append([movie_dict[arr[0]], director_dict[arr[1]]])

    MW = []
    with open('{}{}.txt'.format(path, MW_file), 'r', encoding='UTF-8') as f:
        for line in f:
            arr = line.split()
            MW.append([movie_dict[arr[0]], writer_dict[arr[1]]])

    MA = np.asarray(MA)
    MD = np.asarray(MD)
    MW = np.asarray(MW)

    #--
    #build index for 2hop adjs

    MAi={}
    MDi={}
    MWi={}
    AMi={}
    DMi={}
    WMi={}

    for i in range(MA.shape[0]):
        m=MA[i,0]
        a=MA[i,1]

        if m not in MAi:
            MAi[m]=set()
        if a not in AMi:
            AMi[a]=set()

        MAi[m].add(a)
        AMi[a].add(m)

    for i in range(MD.shape[0]):
        m = MD[i, 0]
        d = MD[i, 1]

        if m not in MDi:
            MDi[m] = set()
        if d not in DMi:
            DMi[d] = set()

        MDi[m].add(d)
        DMi[d].add(m)

    for i in range(MW.shape[0]):
        m = MW[i, 0]
        w = MW[i, 1]

        if m not in MWi:
            MWi[m] = set()
        if w not in WMi:
            WMi[w] = set()

        MWi[m].add(w)
        WMi[w].add(m)

    MAMi={}
    MDMi={}
    MWMi={}

    for v in MAi:
        for a in MAi[v]:
            if a not in AMi:
                continue
            for m in AMi[a]:
                if m not in MAMi:
                    MAMi[m] ={}
                if v not in MAMi:
                    MAMi[v] ={}

                if v not in MAMi[m]:
                    MAMi[m][v]=set()
                if m not in MAMi[v]:
                    MAMi[v][m]=set()

                MAMi[m][v].add(a)
                MAMi[v][m].add(a)

    for v in MDi:
        for d in MDi[v]:
            if d not in DMi:
                continue
            for m in DMi[d]:
                if m not in MDMi:
                    MDMi[m] = {}
                if v not in MDMi:
                    MDMi[v] = {}

                if v not in MDMi[m]:
                    MDMi[m][v] = set()
                if m not in MDMi[v]:
                    MDMi[v][m] = set()

                MDMi[m][v].add(d)
                MDMi[v][m].add(d)

    for v in MWi:
        for w in MWi[v]:
            if w not in WMi:
                continue
            for m in WMi[w]:
                if m not in MWMi:
                    MWMi[m] ={}
                if v not in MWMi:
                    MWMi[v] ={}

                if v not in MWMi[m]:
                    MWMi[m][v]=set()
                if m not in MWMi[v]:
                    MWMi[v][m]=set()

                MWMi[m][v].add(w)
                MWMi[v][m].add(w)


    node_emb, n_nodes, emb_len = read_embed(path=path,emb_file="MADW_16")
    print(n_nodes, emb_len)

    # MAM;
    MAM_emb = []
    for v in MAMi:
        result = {}
        for m in MAMi[v]:
            edge1 = [node_emb[p] for p in MAMi[v][m]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            if m not in result:
                result[m] = edge1
            else:
                result[m] += edge1

        for m in result:
            if v < m:
                MAM_emb.append(np.concatenate(([v, m], result[m])))
    MAM_emb = np.asarray(MAM_emb)
    print("compute edge embeddings {} complete".format('MAM'))

    # MDM;
    MDM_emb = []
    for v in MDMi:
        result = {}
        for m in MDMi[v]:
            edge1 = [node_emb[p] for p in MDMi[v][m]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            if m not in result:
                result[m] = edge1
            else:
                result[m] += edge1

        for m in result:
            if v < m:
                MDM_emb.append(np.concatenate(([v, m], result[m])))
    MDM_emb = np.asarray(MDM_emb)
    print("compute edge embeddings {} complete".format('MDM'))

    # MWM;
    MWM_emb = []
    for v in MWMi:
        result = {}
        for m in MWMi[v]:
            edge1 = [node_emb[p] for p in MWMi[v][m]]
            edge1 = np.sum(np.vstack(edge1), axis=0)  # edge1: the emd between v and a1

            if m not in result:
                result[m] = edge1
            else:
                result[m] += edge1

        for m in result:
            if v < m:
                MWM_emb.append(np.concatenate(([v, m], result[m])))
    MWM_emb = np.asarray(MWM_emb)
    print("compute edge embeddings {} complete".format('MWM'))

    np.savez("{}edge{}.npz".format(path, emb_len),
             MAM=MAM_emb, MDM=MDM_emb, MWM=MWM_emb)
    print('dump npz file {}edge{}.npz complete'.format(path, emb_len))
    pass


# gen_homograph()

dump_yago_edge_emb()