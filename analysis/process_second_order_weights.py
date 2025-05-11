import copy
import os

import numpy as np
from joblib import load
from scipy.io import savemat


def get_2nd_order_weight(file_name, file_dir):
    file_path = os.path.join(file_dir, file_name)
    model = load(file_path)
    return model.coef_


def fetch_weights_joblib(base_dir, task, num_repeat=1000, permutation=False):
    """

    Args:
        base_dir:
        task:
        num_repeat:
        permutation:

    Returns:

    """

    sub_dir = os.path.join(base_dir, task)

    file_name = copy.copy(task)

    if permutation:
        sub_dir = sub_dir + "_permut"
        file_name = file_name + "_permut"

    if not os.path.exists(sub_dir):
        print("No such directory: %s" % sub_dir)
        return None

    weight = []

    for i in range(num_repeat):
        model_file = "%s_%s.joblib" % (file_name, i)
        if os.path.exists(os.path.join(sub_dir, model_file)):
            weight.append(get_2nd_order_weight(model_file, sub_dir).reshape((1, -1)))

    return np.concatenate(weight, axis=0)


def main():
    tasks = [
        # 'L0G1_vs_L2G1',
        "L2G0_vs_L2G1",
        "L0G0_vs_L0G1",
        # 'L0G0_vs_L2G0',
        "L5G0_vs_L5G1",
        "L0G0_vs_L5G0",
        "L0G1_vs_L5G1",
    ]
    base_dirs = {
        "HCP": "/media/shuo/MyDrive/data/HCP/BNA/Models",
        "gsp": "/media/shuo/MyDrive/data/brain/brain_networks/gsp/Models",
    }
    output_dir = "model_weights/second_order"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    n_repeats = 1000

    for dataset in base_dirs.keys():
        base_dir = base_dirs[dataset]
        for task in tasks:
            for permutation in [False, True]:
                print("Processing %s %s permute %s" % (dataset, task, permutation))
                weight = fetch_weights_joblib(base_dir, task, n_repeats, permutation)
                if weight is not None:
                    weight_out = weight.astype(np.float32)
                    w_mean = np.mean(weight, axis=0)
                    w_std = np.std(weight, axis=0)
                    midc = {"mean": w_mean, "std": w_std}
                    fname = "%s_%s" % (dataset, copy.copy(task))
                    if permutation:
                        fname = fname + "_permut"
                    np.savez_compressed(
                        os.path.join(output_dir, fname + ".npz"), weight_out
                    )
                    savemat(os.path.join(output_dir, fname + ".mat"), midc)


if __name__ == "__main__":
    main()
