''' Using database semi-joins to generate list of metapaths
Input: PA, PC
Output: APA.path, APCPA.path, APAPA.path
'''

import numpy as np
import scipy.sparse as sp

from metapath import *

def genPath(path="../data/dblp/"):
    APA_file = "APA"
    APAPA_file = "APAPA"
    APCPA_file = "APCPA"

    adjs, features, labels, idx_train, idx_val, idx_test, node_emb, index \
            = read_mpindex_dblp(path="../data/dblp/")


    APA = index['APA']
    APC = index['APC']
    CPA = index['CPA']

    with open("{}{}.path".format(path, APA_file), mode='w') as f:
        for a1 in APA:
            for a2 in APA[a1]:
                for p in APA[a1][a2]:
                    f.write('{} {} {}\n'.format(a1, p, a2))
    print("dump index {} complete".format(APA_file))

    #APAPA
    with open("{}{}.path".format(path, APAPA_file), mode='w') as f:
        for a1 in APA:
            for a2 in APA[a1]:
                for a3 in APA[a2]:
                    for p1 in APA[a1][a2]:
                        for p2 in APA[a2][a3]:
                            f.write('{} {} {} {} {}\n'.format(a1, p1, a2, p2, a3))
    print("dump index {} complete".format(APAPA_file))

    # APCPA
    with open("{}{}.path".format(path, APCPA_file), mode='w') as f:
        for a1 in APC:
            for c in APC[a1]:
                for a2 in CPA[c]:
                    for p1 in APC[a1][c]:
                        for p2 in CPA[c][a2]:
                            f.write('{} {} {} {} {}\n'.format(a1, p1, c, p2, a2))
    print("dump index {} complete".format(APCPA_file))



    pass

# genPath()



