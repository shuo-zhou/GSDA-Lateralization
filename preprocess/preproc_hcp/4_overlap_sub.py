import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import io_


def main():
    atlas = "BNA"

    data_dir = "../%s/Proc" % atlas
    out_dir = "../Proc/fisherz" % atlas

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    sessions = ["REST1", "REST2"]

    info = dict()
    data = dict()
    run_ = "Fisherz"
    connection_type = "intra"

    for session in sessions:
        info_file = "HCP_%s_half_brain_%s.csv" % (atlas, session)
        info[session] = io_.read_tabular(
            os.path.join(data_dir, info_file), index_col="ID"
        )
        data[session] = io_.load_half_brain(
            data_dir, atlas, session, run_, connection_type
        )

    idx = dict()
    idx["REST1"] = info["REST1"].index.isin(info["REST2"].index)
    idx["REST2"] = info["REST2"].index.isin(info["REST1"].index)

    for session in sessions:
        for side_ in ["Left", "Right"]:
            data[session][side_] = data[session][side_][idx[session]]
        info[session] = info[session][idx[session]]

        out_fname = "HCP_%s_%s_half_brain_%s_%s.hdf5" % (
            atlas,
            connection_type,
            session,
            run_,
        )
        io_.save_half_brain(
            out_dir, out_fname, data[session]["Left"], data[session]["Right"]
        )

    if not np.array_equal(info["REST1"].index.values, info["REST2"].index.values):
        raise ValueError("The subjects id between the two runs are different!")

    info["REST1"].to_csv(os.path.join(out_dir, "HCP_%s_half_brain.csv" % atlas))


if __name__ == "__main__":
    main()
