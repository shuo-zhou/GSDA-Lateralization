import os
import sys

import numpy as np
from sklearn.decomposition import PCA

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io_


def reduce_dim(data, pca):
    data_new = dict()

    for half_ in ["Left", "Right"]:
        data_new[half_] = pca.transform(data[half_])

    return data_new


def main():
    atlas = "BNA"
    data_dir = "/media/shuo/MyDrive/HCP/%s/Proc" % atlas
    out_dir = "/media/shuo/MyDrive/HCP/%s/Proc" % atlas

    session = "REST1"
    connection_type = "intra"
    run_ = "Fisherz"
    n_comp = 30

    data = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)

    pca = PCA(n_components=n_comp, random_state=144)

    pca.fit(np.concatenate((data["Left"], data["Right"])))

    data_new = reduce_dim(data, pca)

    out_fname = "HCP_%s_%s_half_brain_%s_%s_%s.hdf5" % (
        atlas,
        connection_type,
        session,
        "Fisherz_pc",
        n_comp,
    )
    io_.save_half_brain(out_dir, out_fname, data_new["Left"], data_new["Right"])

    session = "REST2"
    data = io_.load_half_brain(data_dir, atlas, session, run_, connection_type)

    data_new = reduce_dim(data, pca)

    out_fname = "HCP_%s_%s_half_brain_%s_%s_%s.hdf5" % (
        atlas,
        connection_type,
        session,
        "Fisherz_pc",
        n_comp,
    )
    io_.save_half_brain(out_dir, out_fname, data_new["Left"], data_new["Right"])


if __name__ == "__main__":
    main()
