import os
import copy
import numpy as np
from joblib import dump, load
from scipy.io import savemat


def get_coef(file_name, file_dir):
    file_path = os.path.join(file_dir, file_name)
    model = load(file_path)
    return model.coef_


def fetch_weights(base_dir, task, permutation=False):
    """

    Args:
        base_dir:
        gender:
        lambda_:

    Returns:

    """

    sub_dir = os.path.join(base_dir, task)
    file_name = copy.copy(task)

    if permutation:
        sub_dir = sub_dir + "_permut"
        file_name = file_name + "_permut"

    weight = []
    num_repeat = 1000

    for i in range(num_repeat):

        model_file = '%s_%s.joblib' % (file_name, i)
        if os.path.exists(os.path.join(sub_dir, model_file)):
            weight.append(get_coef(model_file, sub_dir).reshape((1, -1)))

    return np.concatenate(weight, axis=0)


def main():
    tasks = [
        # 'L0G1_vs_L2G1',
        # 'L2G0_vs_L2G1',
        # 'L0G0_vs_L0G1',
        # 'L0G0_vs_L2G0',
        # 'L5G0_vs_L5G1',
        'L0G0_vs_L5G0',
        'L0G1_vs_L5G1',
    ]
    base_dir = "/media/shuo/MyDrive/data/HCP/BNA/Models"

    for task in tasks:
        for permutation in [False, True]:
            weight = fetch_weights(base_dir, task, permutation)
            w_mean = np.mean(weight, axis=0)
            w_std = np.std(weight, axis=0)
            midc = {"mean": w_mean, "std": w_std}
            fname = copy.copy(task)
            if permutation:
                fname = fname + "_permut"
            savemat(os.path.join(base_dir, fname + ".mat"), midc)


if __name__ == '__main__':
    main()
