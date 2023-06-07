import os

import numpy as np
from scipy.io import savemat
from second_order_clf import fetch_weights


def main():
    dataset = "gsp"
    # base_dir = "/media/shuo/MyDrive/data/brain/brain_networks/ukbio/Models"
    base_dir = "/media/shuo/MyDrive/data/brain/brain_networks/gsp/Models"
    # sessions = ["REST1_", "REST2_"]
    sessions = [""]
    seed_ =2023
    # for lambda_ in [0, 2]:
    # for lambda_ in ["0_mix_gender"]:
    #     for gender in ["mix"]:
    for lambda_ in [0, 2, 5]:
        for gender in [0, 1]:
            weight = fetch_weights(base_dir, gender, lambda_, dataset, sessions=sessions, seed_=seed_)
            w_mean = np.mean(weight, axis=0)
            w_std = np.std(weight, axis=0)
            midc = {"mean": w_mean, "std": w_std}
            fname = "L%sG%s.mat" % (lambda_, gender)
            savemat(os.path.join(base_dir, fname + ".mat"), midc)


if __name__ == '__main__':
    main()
