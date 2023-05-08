import os
import sys
from scipy.io import loadmat

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io_


# basedir = 'D:/ShareFolder/AICHA_VolFC'  # 'D:\ShareFolder\AICHA_VolFC\REST2\LR\FC_R'
atlas = 'BNA'
basedir = 'C:/Data/brain/validation/ukbio/'
sub_info_fpath = 'C:/Data/brain/validation/ukbio/infoYLY.mat'
out_dir = os.path.join(basedir, "proc")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

connection_type = 'intra'  # inter & intra
# n_comps = [50, 10, 150, 200, 250]

# sub_info = io_.read_table(os.path.join(basedir, sub_info_fname), index_col=0)
sub_info = loadmat(sub_info_fpath)["info"]
sub_idx = sub_info[:, 0]


fpath_df = io_.get_fpaths(os.path.join(basedir, atlas), sub_idx)
fpaths = fpath_df['File path']
data_left, data_right = io_.load_txt(fpaths, connection_type=connection_type)
out_fname = 'ukbio_%s_%s_half_brain.hdf5' % (atlas, connection_type)
io_.save_half_brain(out_dir, out_fname, data_left, data_right)
