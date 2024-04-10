import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import io_


def main():
    atlas = "BNA"
    data_dir = "../%s/Proc" % atlas
    out_dir = "../Result" % atlas

    sessions = ["REST1", "REST2"]

    data = dict()
    run_ = "Fisherz"
    connection_type = "intra"

    info_file = "HCP_%s_half_brain.csv" % atlas
    info = io_.read_tabular(os.path.join(data_dir, info_file), index_col="ID")

    for session in sessions:
        data[session] = io_.load_half_brain(
            data_dir, atlas, session, run_, connection_type
        )

    gender = info["gender"].values

    idx_male = np.where(gender == 0)[0]
    idx_female = np.where(gender == 1)[0]
    np.random.seed(144)
    idx_female_rand = np.random.choice(
        idx_female, size=idx_male.shape[0], replace=False
    )
    idx_female_rand.sort()
    idx = np.concatenate((idx_male, idx_female_rand))
    idx.sort()

    info = info.iloc[idx]

    for session in sessions:
        for side_ in ["Left", "Right"]:
            data[session][side_] = data[session][side_][idx]

        out_fname = "HCP_%s_%s_half_brain_%s_gender_equal_%s.hdf5" % (
            atlas,
            connection_type,
            session,
            run_,
        )
        io_.save_half_brain(
            out_dir, out_fname, data[session]["Left"], data[session]["Right"]
        )

    info.to_csv(os.path.join(out_dir, "HCP_%s_half_brain_gender_equal.csv" % atlas))


if __name__ == "__main__":
    main()
