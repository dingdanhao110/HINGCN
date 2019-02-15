from utilities import *
from scipy import sparse as sp


def gen_metapath_dblp_store(path="/home/danhao/Git/gcn/HINGCN/trunk/data/dblp/"):
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


gen_metapath_dblp_store()
