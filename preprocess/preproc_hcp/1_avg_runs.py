import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import io_


def main():
    atlas = "BNA"

    data_dir = "../%s/Proc" % atlas
    out_dir = "../%s/Proc" % atlas

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    session = "REST1"
    runs = ["LR", "RL"]
    info = dict()
    data = dict()

    connection_type = "intra"

    for run_ in runs:
        info_file = "HCP_%s_half_brain_%s_%s.csv" % (atlas, session, run_)
        info[run_] = io_.read_tabular(os.path.join(data_dir, info_file), index_col="ID")
        data[run_] = io_.load_half_brain(
            data_dir, atlas, session, run_, connection_type
        )

    idx = dict()
    idx["LR"] = info["LR"].index.isin(info["RL"].index)
    idx["RL"] = info["RL"].index.isin(info["LR"].index)

    for run_ in runs:
        for side_ in ["Left", "Right"]:
            data[run_][side_] = data[run_][side_][idx[run_]]
        info[run_] = info[run_][idx[run_]]

    if not np.array_equal(info["LR"].index.values, info["RL"].index.values):
        raise ValueError("The subjects id between the two runs are different!")

    data_left = (data["LR"]["Left"] + data["RL"]["Left"]) / 2
    data_right = (data["LR"]["Right"] + data["RL"]["Right"]) / 2

    out_fname = "HCP_%s_%s_half_brain_%s_AVG.hdf5" % (atlas, connection_type, session)
    io_.save_half_brain(out_dir, out_fname, data_left, data_right)
    info["LR"].to_csv(
        os.path.join(out_dir, "HCP_%s_half_brain_%s.csv" % (atlas, session))
    )


if __name__ == "__main__":
    main()
