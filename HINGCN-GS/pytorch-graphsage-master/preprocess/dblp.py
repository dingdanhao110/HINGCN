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

clean_dblp()
gen_homograph()