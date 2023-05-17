import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io_


# basedir = 'D:/ShareFolder/AICHA_VolFC'  # 'D:\ShareFolder\AICHA_VolFC\REST2\LR\FC_R'
atlas = 'BNA'
basedir = '/media/shuo/MyDrive/data/HCP/hcp-Retest'
sub_info_fpath = '/media/shuo/MyDrive/data/HCP/AICHA/AtlasInfo/AtlasInfo/HCP_basic_information.xlsx'
out_dir = '/media/shuo/MyDrive/data/HCP/hcp-Retest/BNA/'
runs = ['LR', 'RL']
sessions = ['REST1', 'REST2']
# sessions = ['REST2']
# out_dir = 'D:/ShareFolder/AICHA_VolFC/Proc'
connection_type = 'intra'  # inter & intra
# n_comps = [50, 10, 150, 200, 250]

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# sub_info = io_.read_table(os.path.join(basedir, sub_info_fname), index_col=0)
sub_info = io_.read_table(sub_info_fpath, index_col=0)
sub_idx = sub_info.index

for session in sessions:
    run_path = os.path.join("%s/BNA" %basedir, session)

    for run in runs:
        scan_path = os.path.join(run_path, '%s' % run)
        fpath_df = io_.get_fpaths(scan_path, sub_idx)
        # idx_valid = fpath_df.index
        # for col in sub_info.columns:
        #     fpath_df[col] = sub_info[col].loc[idx_valid]
        # fpath_df.to_csv(os.path.join(out_dir, 'HCP_%s_half_brain_%s_%s.csv' % (atlas, session, run)),
        #                 index_label='ID')
        fpaths = fpath_df['File path']
        data_left, data_right = io_.load_txt(fpaths, connection_type=connection_type)
        out_fname = 'HCP_%s_%s_half_brain_%s_%s.hdf5' % (atlas, connection_type, session, run)
        io_.save_half_brain(out_dir, out_fname, data_left, data_right)

        df_out = pd.concat((fpath_df, sub_info.loc[fpath_df.index.values, :]), axis=1)
        df_out.to_csv(os.path.join(out_dir, 'HCP_%s_half_brain_%s_%s.csv' % (atlas, session, run)))
