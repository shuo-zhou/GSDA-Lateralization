"""
Author: Shuo Zhou

This script is used to combine the first-order coefficients of over multiple model files with different train test 
splits and save them in a single .npz file for each experimental setting.

The script fetches the weights from the model files, computes the mean and standard deviation of the weights, and saves 
them in .mat format.
"""

import os

import numpy as np
from scipy.io import savemat
from scipy.stats import normaltest
from second_order_clf import fetch_weights


def main():
    # base_dir = "./output"
    output_dir = "first-order_coef"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    seed_ = 2023
    base_dirs = {
        "gsp": {"00": ([""], "/media/shuo/MyDrive/data/brain/brain_networks/gsp/Models")},
        "HCP": {"00": (["REST1_", "REST2_"], "/media/shuo/MyDrive/data/HCP/BNA/Models"),
                "02":  ([""], "/media/shuo/MyDrive/data/HCP/BNA/Models/test_size02")}
    }

    for dataset in base_dirs.keys():
        for test_size in base_dirs[dataset].keys():
            sessions = base_dirs[dataset][test_size][0]
            base_dir = base_dirs[dataset][test_size][1]
            for lambda_ in ["0_group_mix", 0, 1, 2, 5, 8, 10]:
                for group in [0, 1]:
                    print("dataset: %s, test_size: %s, lambda_: %s, group: %s" % (dataset, test_size, lambda_, group))
                    weight = fetch_weights(
                        base_dir, group, lambda_, dataset, sessions=sessions, test_size=test_size, seed_=seed_
                    )
                    if weight is not None:
                        p_vals = []
                        for _ in range(weight.shape[0]):
                            p_vals.append(normaltest(weight[_])[1])
                        weight_out = weight.astype(np.float32)
                        w_mean = np.mean(weight, axis=0)
                        w_std = np.std(weight, axis=0)
                        midc = {"mean": w_mean, "std": w_std}
                        fname = "%s_L%sG%s_test_size%s" % (dataset, lambda_, group, test_size)
                        np.savez_compressed(os.path.join(output_dir, fname + ".npz"), weight_out)
                        savemat(os.path.join(output_dir, fname + ".mat"), midc)


if __name__ == "__main__":
    main()
