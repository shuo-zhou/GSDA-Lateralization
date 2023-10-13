import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io_


def main():
    atlas = 'BNA'
    # data_dir = "/media/shuo/MyDrive/HCP/%s/Proc" % atlas
    # out_dir = '/media/shuo/MyDrive/HCP/%s/Proc' % atlas

    data_dir = "/media/shuo/MyDrive/data/HCP/hcp-Retest/%s/Proc" % atlas
    out_dir = '/media/shuo/MyDrive/data/HCP/hcp-Retest/%s/Proc' % atlas

    session = 'REST2'

    connection_type = 'intra'

    data = io_.load_half_brain(data_dir, atlas, session, 'AVG', connection_type)

    for half_ in ['Left', 'Right']:
        data[half_] = np.arctanh(data[half_])

    out_fname = 'HCP_%s_%s_half_brain_%s_%s.hdf5' % (atlas, connection_type, session, 'Fisherz')
    io_.save_half_brain(out_dir, out_fname, data['Left'], data['Right'])


if __name__ == '__main__':
    main()
