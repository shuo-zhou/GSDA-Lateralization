import os

import numpy as np
import torch
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle

# from skops.io import dump


def get_coef(file_name, file_dir):
    file_path = os.path.join(file_dir, file_name)
    model = torch.load(file_path)
    return model.theta


# def get_model_weight(file_name, file_dir):
#     file_path = os.path.join(file_dir, file_name)
#     model = torch.load(file_path)
#     return model.model.weight.data.numpy().T


def fetch_weights(base_dir, gender, lambda_, dataset, sessions, seed_=2023):
    """

    Args:
        base_dir:
        gender:
        lambda_:

    Returns:
        a matrix of weights, shape (n_models, n_features)
    """

    sub_dir = os.path.join(base_dir, "lambda%s" % lambda_)
    if lambda_ == "0_mix_gender":
        lambda_ = 0
    weight = []
    num_repeat = 5
    halfs = [0, 1]
    for session_i in sessions:
        for half_i in halfs:
            for i_split in range(num_repeat):
                for seed in range(50):
                    model_file = "%s_lambda%s_%s%s_%s_gender_%s_%s.pt" % (
                        dataset,
                        lambda_,
                        session_i,
                        i_split,
                        half_i,
                        gender,
                        seed_ - seed,
                    )
                    if os.path.exists(os.path.join(sub_dir, model_file)):
                        weight.append(get_coef(model_file, sub_dir).reshape((1, -1)))

    return np.concatenate(weight, axis=0)


def main():
    model1 = {"lambda": 2, "gender": 0}
    model2 = {"lambda": 2, "gender": 1}

    # base_dir = "/media/shuo/MyDrive/data/HCP/BNA/Models"
    # base_dir = "/media/shuo/MyDrive/data/brain/brain_networks/gsp/Models"
    # dataset = "gsp"
    base_dir = "/media/shuo/MyDrive/data/brain/brain_networks/ukbio/Models"
    dataset = "ukb"
    sessions = [""]
    seed_ = 2023

    permutation = False
    num_splits = 1000
    random_state = 2023

    weights1 = fetch_weights(
        base_dir,
        model1["gender"],
        model1["lambda"],
        dataset,
        sessions=sessions,
        seed_=seed_,
    )
    weights2 = fetch_weights(
        base_dir,
        model2["gender"],
        model2["lambda"],
        dataset,
        sessions=sessions,
        seed_=seed_,
    )

    weights = np.concatenate((weights1, weights2), axis=0)
    labels = np.zeros(weights.shape[0])
    labels[: weights1.shape[0]] = 1

    res = {"accuracy": []}
    # acc = []
    i_iter = 0

    task = "L%sG%s_vs_L%sG%s" % (
        model1["lambda"],
        model1["gender"],
        model2["lambda"],
        model2["gender"],
    )
    if permutation:
        labels = shuffle(labels, random_state=random_state)
        task = task + "_permut"
    out_dir = os.path.join(base_dir, task)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sss = StratifiedShuffleSplit(
        n_splits=num_splits, test_size=0.2, random_state=random_state
    )
    for train, test in sss.split(weights, labels):
        clf = LogisticRegression()
        clf.fit(weights[train], labels[train])

        acc = accuracy_score(labels[test], clf.predict(weights[test]))
        res["accuracy"].append(acc)

        model_fname = "%s_%s.joblib" % (task, i_iter)
        # model_fname = "%s_%s.skops" % (task, i_iter)
        i_iter += 1
        dump(clf, os.path.join(out_dir, model_fname))

    print(np.max(res["accuracy"]), np.min(res["accuracy"]))
    print(np.mean(res["accuracy"]))
    print(np.std(res["accuracy"]))
    print("Done")


if __name__ == "__main__":
    main()
