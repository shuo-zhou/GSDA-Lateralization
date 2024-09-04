import os

import numpy as np
from scipy.io import savemat
from scipy.stats import normaltest
from second_order_clf import fetch_weights


def main():
    dataset = "HCP"
    base_dir = "./output"
    output_dir = "first-order/HCP"
    seed_ = 2022
    sessions = ["REST1_", "REST2_"]

    for lambda_ in [0, 1, 2, 5, 8, 10]:
        # for lambda_ in [1, 8, 10]:
        # for lambda_ in ["0_mix_gender"]:
        #     for gender in ["mix"]:
        # for lambda_ in [0, 2, 5]:
        for gender in [0, 1]:
            weight = fetch_weights(
                base_dir, gender, lambda_, dataset, sessions=sessions, seed_=seed_
            )
            p_vals = []
            for _ in range(weight.shape[0]):
                p_vals.append(normaltest(weight[_])[1])
            w_mean = np.mean(weight, axis=0)
            w_std = np.std(weight, axis=0)
            midc = {"mean": w_mean, "std": w_std}
            fname = "%s_L%sG%s" % (dataset, lambda_, gender)
            savemat(os.path.join(output_dir, fname + ".mat"), midc)


if __name__ == "__main__":
    main()
