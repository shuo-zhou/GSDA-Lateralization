import os
import sys
import h5py
import numpy as np
import pandas as pd


def load_txt(fpaths, connection_type='intra'):
    """Load data from a list of txt files。

    Parameters
    ----------
    fpaths (list): A list of .txt file paths.

    Returns
    -------
        Two ndarrays
    """
    left_ = []
    right_ = []
    for fpath in fpaths:
        data_matrix = np.genfromtxt(fpath)
        left_vec, right_vec = split_brain(data_matrix, connection_type=connection_type)
        left_.append(left_vec.reshape((1, -1)))
        right_.append(right_vec.reshape((1, -1)))
    left_ = np.concatenate(left_, axis=0)
    right_ = np.concatenate(right_, axis=0)

    return left_, right_


def load_hdf5(fpath):
    f = h5py.File(fpath, 'r')
    data = {'Left': f['Left'][()],
            'Right': f['Right'][()]}
    return data


def read_table(fname, sheet_name=0, index_col=0):
    """Read a table from a .xlsx or .csv file

    Parameters
    ----------
    fname (string):
    sheet_name
    index_col

    Returns
    -------

    """
    file_format = fname.split('.')[-1]
    if file_format == 'xlsx':
        df = pd.read_excel(fname, sheet_name=sheet_name,
                           index_col=index_col, engine='openpyxl')
    elif file_format == 'csv':
        df = pd.read_csv(fname, index_col=index_col)

    return df


def get_fpaths(fdir, idx_list, file_format='txt'):
    """Check the existence of files named by subject indices under a given directory

    Parameters
    ----------
    fdir (string): file directory for data
    idx_list (list): a list of subject indices
    file_format (string): file format, defaults to txt

    Returns
    -------
        Dataframe
    """
    fpaths = dict()
    for idx in idx_list:
        fname = '%s.%s' % (idx, file_format)
        fpath = os.path.join(fdir, fname)
        if os.path.exists(fpath):
            fpaths[idx] = fpath

    fpath_df = pd.DataFrame(data={'File path': fpaths.values()}, index=fpaths.keys())

    return fpath_df


def split_brain(matrix, connection_type='intra'):
    # left_vec = []
    # right_vec = []
    # n_roi = matrix.shape[0]
    # for i in range(n_roi):
    #     if i % 2 == 1:
    #         right_vec.append(matrix[i, :].reshape(1, -1))
    #     else:
    #         left_vec.append(matrix[i, :].reshape(1, -1))
    #
    # left_vec = left.reshape((1, -1))
    # right_vec = right.reshape((1, -1))
    n_ = matrix.shape[1] / 2
    if connection_type == 'intra':
        left = matrix[0::2, 0::2]  # even indices of rows, left brain
        right = matrix[1::2, 1::2]  # odd indices of rows, right brain

        idx = np.triu_indices(n_, k=1)
        n_feat = int(n_ * (n_ - 1) / 2)
        left_vec = np.zeros(n_feat)
        right_vec = np.zeros(n_feat)

        left_vec[:] = left[idx]
        right_vec[:] = right[idx]
    elif connection_type == 'inter':
        left = matrix[0::2, 1::2]
        right = matrix[1::2, 2::2]

        left_vec = left.reshape((1, -1))
        right_vec = right.reshape((1, -1))

    return left_vec, right_vec


def save_half_brain(out_dir, out_fname, data_left, data_right):
    """Save two hemisphere nadarrays into an hdf5 file

    Parameters
    ----------
    out_dir (string):
    out_fname (string):
    data_left (ndarray):
    data_right (ndarray):

    Returns
    -------

    """
    f = h5py.File(os.path.join(out_dir, out_fname), "w")
    f.create_dataset('Left', data=data_left)
    f.create_dataset('Right', data=data_right)
    f.close()
