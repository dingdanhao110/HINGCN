from utilities import *
from scipy import sparse as sp


def gen_metapath_dblp_store(path="./data/dblp/"):
    PA_file = "PA"
    PC_file = "PC"
    PT_file = "PT"

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
    # build graph
    APA = PA.transpose() * PA
    APAPA = PA.transpose() * PA * PA.transpose() * PA
    APCPA = PA.transpose() * PC * PC.transpose() * PA

    APA = pathsim(APA)
    print('finished APA')
    APAPA = pathsim(APAPA)
    print('finished APAPA')
    APCPA = pathsim(APCPA)
    print('finished APCPA')

    # To Dump: pathsim of APA, APAPA, APCPA
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    sp.save_npz("{}{}.npz".format(path, APA_file), APA)
    sp.save_npz("{}{}.npz".format(path, APAPA_file), APAPA)
    sp.save_npz("{}{}.npz".format(path, APCPA_file), APCPA)

    return


def gen_metapath_dblp_cnt(path="./data/dblp/"):
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

    paper_max = max(PA[:, 0]) + 1
    author_max = max(PA[:, 1]) + 1
    conf_max = max(PC[:, 1]) + 1

    PA = sp.coo_matrix((np.ones(PA.shape[0]), (PA[:, 0], PA[:, 1])),
                       shape=(paper_max, author_max),
                       dtype=np.int32)
    PC = sp.coo_matrix((np.ones(PC.shape[0]), (PC[:, 0], PC[:, 1])),
                       shape=(paper_max, conf_max),
                       dtype=np.int32)
    # build graph
    APA = PA.transpose() * PA
    APAPA = PA.transpose() * PA * PA.transpose() * PA
    APCPA = PA.transpose() * PC * PC.transpose() * PA

    # To Dump: cnt of APA, APAPA, APCPA
    APA_file = "APA_cnt"
    APAPA_file = "APAPA_cnt"
    APCPA_file = "APCPA_cnt"

    sp.save_npz("{}{}.npz".format(path, APA_file), APA)
    sp.save_npz("{}{}.npz".format(path, APAPA_file), APAPA)
    sp.save_npz("{}{}.npz".format(path, APCPA_file), APCPA)

    return


def gen_metapath_yago_cnt(path="./data/freebase/"):
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
    actor_dict = {a: i  for (i, a) in enumerate(actors)}
    director_dict = {a: i  for (i, a) in enumerate(directors)}
    writer_dict = {a: i  for (i, a) in enumerate(writers)}

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


    MA = sp.coo_matrix((np.ones(MA.shape[0]), (MA[:, 0], MA[:, 1])),
                       shape=(n_movie, n_actor),
                       dtype=np.int32)
    MD = sp.coo_matrix((np.ones(MD.shape[0]), (MD[:, 0], MD[:, 1])),
                       shape=(n_movie, n_director),
                       dtype=np.int32)
    MW = sp.coo_matrix((np.ones(MW.shape[0]), (MW[:, 0], MW[:, 1])),
                       shape=(n_movie, n_writer),
                       dtype=np.int32)
    # build graph
    MAM = MA * MA.transpose()
    MAM.setdiag(0)
    MDM = MD * MD.transpose()
    MDM.setdiag(0)
    MWM = MW * MW.transpose()
    MWM.setdiag(0)

    # To Dump: cnt of APA, APAPA, APCPA
    MAM_file = "MAM_cnt"
    MDM_file = "MDM_cnt"
    MWM_file = "MWM_cnt"

    sp.save_npz("{}{}.npz".format(path, MAM_file), MAM)
    sp.save_npz("{}{}.npz".format(path, MDM_file), MDM)
    sp.save_npz("{}{}.npz".format(path, MWM_file), MWM)

    return


def gen_metapath_yelp_cnt(path="./data/yelp/"):
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

    rate_max = max(RB[:, 0]) + 1  # 33360
    busi_max = max(RB[:, 1]) + 1  # 2614
    key_max = max(RK[:, 1]) + 1  # 82
    user_max = max(RU[:, 1]) + 1  # 1286

    RB = sp.coo_matrix((np.ones(RB.shape[0]), (RB[:, 0], RB[:, 1])),
                       shape=(rate_max, busi_max),
                       dtype=np.int32)
    RK = sp.coo_matrix((np.ones(RK.shape[0]), (RK[:, 0], RK[:, 1])),
                       shape=(rate_max, key_max),
                       dtype=np.int32)
    RU = sp.coo_matrix((np.ones(RU.shape[0]), (RU[:, 0], RU[:, 1])),
                       shape=(rate_max, user_max),
                       dtype=np.int32)

    # build graph
    BRURB= RB.transpose() *RU * RU.transpose() * RB
    BRURB.setdiag(0)
    BRKRB = RB.transpose() *RK * RK.transpose() * RB
    BRKRB.setdiag(0)

    # To Dump: cnt of APA, APAPA, APCPA
    BRURB_file = "BRURB_cnt"
    BRKRB_file = "BRKRB_cnt"

    sp.save_npz("{}{}.npz".format(path, BRURB_file), BRURB)
    sp.save_npz("{}{}.npz".format(path, BRKRB_file), BRKRB)

    return

gen_metapath_yelp_cnt()
# gen_metapath_dblp_cnt()
