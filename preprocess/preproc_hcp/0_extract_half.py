import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import io_

atlas = "BNA"
basedir = ""
sub_info_fpath = ""
out_dir = ""
runs = ["LR", "RL"]
sessions = ["REST1", "REST2"]

connection_type = "intra"  # inter & intra

sub_info = io_.read_tabular(sub_info_fpath, index_col=0)
sub_idx = sub_info.index

for session in sessions:
    run_path = os.path.join(basedir, session)

    for run in runs:
        scan_path = os.path.join(run_path, "%s/FC_R" % run)
        fpath_df = io_.get_fpaths(scan_path, sub_idx)
        # idx_valid = fpath_df.index
        # for col in sub_info.columns:
        #     fpath_df[col] = sub_info[col].loc[idx_valid]
        # fpath_df.to_csv(os.path.join(out_dir, 'HCP_%s_half_brain_%s_%s.csv' % (atlas, session, run)),
        #                 index_label='ID')
        fpaths = fpath_df["File path"]
        data_left, data_right = io_.load_txt(fpaths, connection_type=connection_type)
        out_fname = "HCP_%s_%s_half_brain_%s_%s.hdf5" % (
            atlas,
            connection_type,
            session,
            run,
        )
        io_.save_half_brain(out_dir, out_fname, data_left, data_right)
