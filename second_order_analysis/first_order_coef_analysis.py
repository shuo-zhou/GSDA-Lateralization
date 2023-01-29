import os
import numpy as np
from second_order_clf import fetch_weights
from scipy.io import savemat


def main():
    base_dir = "/media/shuo/MyDrive/data/HCP/BNA/Models"
    # for lambda_ in [0, 2]:
    for lambda_ in [5]:
        for gender in [0, 1]:
            weight = fetch_weights(base_dir, gender, lambda_)
            w_mean = np.mean(weight, axis=0)
            w_std = np.std(weight, axis=0)
            midc = {"mean": w_mean, "std": w_std}
            fname = "L%sG%s.mat" % (lambda_, gender)
            savemat(os.path.join(base_dir, fname + ".mat"), midc)


if __name__ == '__main__':
    main()
