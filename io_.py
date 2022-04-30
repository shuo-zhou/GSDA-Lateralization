import os
import h5py
import numpy as np
import pandas as pd
from scipy.io import loadmat


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
        left_vec, right_vec = split_functional_brain(data_matrix, connection_type=connection_type)
        left_.append(left_vec.reshape((1, -1)))
        right_.append(right_vec.reshape((1, -1)))
    left_ = np.concatenate(left_, axis=0)
    right_ = np.concatenate(right_, axis=0)

    return left_, right_


def load_hdf5(fpath):
    """

    Parameters
    ----------
    fpath

    Returns
    -------

    """
    f = h5py.File(fpath, 'r')
    data = {'Left': f['Left'][()],
            'Right': f['Right'][()]}
    return data


def read_table(fname, **kwargs):
    """Read a table from a .xlsx or .csv file

    Parameters
    ----------
    fname (string):

    Returns
    -------

    """
    file_format = fname.split('.')[-1]
    if file_format == 'xlsx':
        df = pd.read_excel(fname, engine='openpyxl', **kwargs)
    elif file_format == 'csv':
        df = pd.read_csv(fname, **kwargs)
    else:
        raise ValueError('Unsupported file type %s' % file_format)

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


def split_functional_brain(matrix, connection_type='intra'):
    """

    Parameters
    ----------
    matrix
    connection_type

    Returns
    -------

    """
    n_ = matrix.shape[1] / 2
    if connection_type == 'intra':
        left = matrix[0::2, 0::2]  # even indices of rows, left brain
        right = matrix[1::2, 1::2]  # odd indices of rows, right brain

        idx = np.triu_indices(n_, k=1)
        n_feat = int(n_ * (n_ - 1) / 2)
        left_vec = np.zeros((1, n_feat))
        right_vec = np.zeros((1, n_feat))
        left_vec[0, :] = left[idx]
        right_vec[0, :] = right[idx]

    elif connection_type == 'inter':
        left = matrix[0::2, 1::2]
        right = matrix[1::2, 0::2]
        left_vec = left.reshape((1, -1))
        right_vec = right.reshape((1, -1))

    else:
        raise ValueError('Invalid connection type %s' % connection_type)

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


def load_half_brain(datadir, atlas, session=None, run=None, connection_type=None, data_type='functional'):
    """

    Parameters
    ----------
    datadir
    atlas
    session
    run
    connection_type
    data_type

    Returns
    -------

    """
    if data_type == 'functional':
        if connection_type == 'both':
            data = {'Left': [], 'Right': []}
            for type_ in ['inter', 'intra']:
                fname = 'HCP_%s_%s_half_brain_%s_%s.hdf5' % (atlas, type_, session, run)
                data_in = load_hdf5(os.path.join(datadir, fname))
                data['Left'].append(data_in['Left'])
                data['Right'].append(data_in['Right'])
            data['Left'] = np.concatenate(data['Left'], axis=1)
            data['Right'] = np.concatenate(data['Right'], axis=1)
        else:
            fname = 'HCP_%s_%s_half_brain_%s_%s.hdf5' % (atlas, connection_type, session, run)
            data = load_hdf5(os.path.join(datadir, fname))
    elif data_type == 'structural':
        data = {'Left': [], 'Right': []}
        fname = '%s_Volume.mat' % atlas
        data_in = loadmat(os.path.join(datadir, fname))['%s_Volume' % atlas][0][0][0]
        data['Left'] = data_in[:, 0::2]
        data['Right'] = data_in[:, 1::2]
    else:
        raise ValueError('Invalid data type %s' % data_type)

    return data
