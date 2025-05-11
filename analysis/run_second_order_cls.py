import os

import numpy as np
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import shuffle


def main():
    model1 = {"lambda": 2, "gender": 0}
    model2 = {"lambda": 2, "gender": 1}

    base_dir = "./model_weights/first_order"
    dataset = "HCP"
    test_size = "00"
    permutation = False
    num_splits = 1000
    random_state = 2023

    weight1_file = os.path.join(
        base_dir,
        "%s_L%dG%d_test_size%s.npz"
        % (dataset, model1["lambda"], model1["gender"], test_size),
    )
    weight2_file = os.path.join(
        base_dir,
        "%s_L%dG%d_test_size%s.npz"
        % (dataset, model2["lambda"], model2["gender"], test_size),
    )
    weights1 = np.load(weight1_file)["arr_0"]
    weights2 = np.load(weight2_file)["arr_0"]

    weights = np.concatenate((weights1, weights2), axis=0)
    labels = np.zeros(weights.shape[0])
    labels[: weights1.shape[0]] = 1

    res = {"accuracy": []}

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

        i_iter += 1
        dump(clf, os.path.join(out_dir, model_fname))

    print(np.max(res["accuracy"]), np.min(res["accuracy"]))
    print(np.mean(res["accuracy"]))
    print(np.std(res["accuracy"]))
    print("Done")


if __name__ == "__main__":
    main()
