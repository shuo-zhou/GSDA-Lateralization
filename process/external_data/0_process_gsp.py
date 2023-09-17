import os
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io_

# basedir = 'D:/ShareFolder/AICHA_VolFC'  # 'D:\ShareFolder\AICHA_VolFC\REST2\LR\FC_R'
atlas = "BNA"
basedir = "/media/shuo/MyDrive/data/brain/brain_networks/gsp/"  # 'C:/Data/brain/validation/ukbio/'
sub_info_fpath = "/media/shuo/MyDrive/data/brain/brain_networks/gsp/participants.tsv"  # 'C:/Data/brain/validation/ukbio/infoYLY.mat'
out_dir = os.path.join(basedir, "proc")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

connection_type = "intra"  # inter & intra
# n_comps = [50, 10, 150, 200, 250]

# sub_info = io_.read_table(os.path.join(basedir, sub_info_fname), index_col=0)
sub_info_df = pd.read_csv(sub_info_fpath, sep="\t")
sub_idx = sub_info_df["participant_id"]

fpath_df = io_.get_fpaths(os.path.join(basedir, atlas), sub_idx)
fpaths = fpath_df["File path"]
data_left, data_right = io_.load_txt(fpaths, connection_type=connection_type)
data_left = np.arctanh(data_left)
data_right = np.arctanh(data_right)

out_fname = "gsp_%s_%s_half_brain_%s.mat" % (atlas, connection_type, "Fisherz")
io_.save_half_brain_mat(out_dir, out_fname, data_left, data_right)

info_out = {
    "ID": fpath_df.index.values,
    "File path": fpaths.values,
    "age": [],
    "gender": [],
}
for sub_id in info_out["ID"]:
    sub_info = sub_info_df[sub_info_df["participant_id"] == sub_id]
    info_out["age"].append(sub_info["age"].values[0])
    if sub_info["sex"].values[0] == "F":
        info_out["gender"].append(1)
    else:
        info_out["gender"].append(0)

info_out_df = pd.DataFrame(info_out)
info_out_df.to_csv(os.path.join(out_dir, "gsp_%s_half_brain.csv" % atlas), index=False)
