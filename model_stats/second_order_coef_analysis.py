import copy
import os
import sys

import numpy as np
from joblib import dump, load
from scipy.io import savemat

sys.path.append('../')
from io_ import fetch_weights_joblib


def main():
    tasks = [
        # 'L0G1_vs_L2G1',
        'L2G0_vs_L2G1',
        'L0G0_vs_L0G1',
        # 'L0G0_vs_L2G0',
        'L5G0_vs_L5G1',
        'L0G0_vs_L5G0',
        'L0G1_vs_L5G1',
    ]
    # base_dir = "/media/shuo/MyDrive/data/HCP/BNA/Models"
    # base_dir = "/media/shuo/MyDrive/data/brain/brain_networks/ukbio/Models"
    base_dir = "/media/shuo/MyDrive/data/brain/brain_networks/gsp/Models"
    n_repeats = 1000

    for task in tasks:
        # for permutation in [False, True]:
        for permutation in [False]:
            weight = fetch_weights_joblib(base_dir, task, n_repeats, permutation)
            w_mean = np.mean(weight, axis=0)
            w_std = np.std(weight, axis=0)
            midc = {"mean": w_mean, "std": w_std}
            fname = copy.copy(task)
            if permutation:
                fname = fname + "_permut"
            savemat(os.path.join(base_dir, fname + ".mat"), midc)


if __name__ == '__main__':
    main()
