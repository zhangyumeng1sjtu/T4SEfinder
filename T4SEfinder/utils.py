import random
from collections import defaultdict

import numpy as np
import torch

from sklearn.preprocessing import minmax_scale


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_pssm(file, length=None):

    PSSM = []
    AAs = 'ARNDCQEGHILKMFPSTWYV'
    aaPSSM = defaultdict(list)
    output = {}
    seq_length = 0

    for line_num, line in enumerate(open(file, "r")):
        if line_num > 2:
            row = line.split()[1:22]
            if len(row) == 0:
                break
            if isinstance(length, int):
                PSSM.append(list(map(int, row[1:])))
            else:
                aaPSSM[row[0]].append(list(map(int, row[1:])))
            seq_length += 1
            
    if isinstance(length, int):
        oriPSSM = np.array(PSSM)
        oriPSSM = minmax_scale(oriPSSM, axis=1)
        output["PSSM"+str(length)] = get_C_terminal(oriPSSM, seq_length, length)

    else:
        maaPSSM = np.zeros((20, 20))
        for i, aa in enumerate(AAs):
            if aa in aaPSSM.keys():
                maaPSSM[i, :] = np.array(aaPSSM[aa]).sum(axis=0)/seq_length   
        output["MaaPSSM"] = maaPSSM

    return output


def get_C_terminal(pssm, seq_length, max_length):

    if seq_length < max_length:
        new_pssm = np.vstack((np.zeros((max_length-seq_length,20)), pssm)) 
    else:
        new_pssm = pssm
    return new_pssm[-max_length:, :]
