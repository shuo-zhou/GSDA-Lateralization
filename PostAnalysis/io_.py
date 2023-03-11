# MRI related func
import pandas as pd
import scipy.io as sio
import numpy as np
import os
import sys
import glob
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import nibabel as nib
from scipy.stats import pearsonr
# read: nib.load()
# write: nib.save()
#### ******************************* file split ********************************** ####
def file_split(file_path):
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension

#### ******************************* read h5 or pickle file as a df ********************************** ####

def read_file(file):
    _, _, ext = file_split(file)
    if ext == '.h5':
        df = pd.read_hdf(file)
    elif ext == '.pkl':
        df = pd.read_pickle(file)
    else:
        print('file shoud be xxx.h5 or xxx.pkl...')

    return df

def select_data_subj(df, subj_id, df_column_name):
    data = df[df_column_name][df['subject'] == subj_id]
    return data # series datatype with dim of (1,),for the ndarray contents,data = data[0]

def select_data_multi_subj(df, subj_ids, df_column_name):
    subj_df = df['subject'].to_numpy()
    sub, index1, index2 = np.intersect1d(subj_df, subj_ids, assume_unique=False, return_indices=True)
    data = df.loc[index1, df_column_name] # loc,select column name. iloc,select data as index
    return data.to_numpy() # convert pd series to numpy ndarray


#### ******************************* read MRI files ****************************** ####
def read_surf(surface):
    surf = nib.load(surface)
    arr = surf.darrays
    coord = arr[0].data # coordinate of vertices
    vert = arr[1].data # index of vertices
    return coord, vert

def read_nii(T1w):
    T1 = nib.load(T1w)
    data = T1.get_fdata() # 3-D ‘ray
    header = T1.header # header
    return data, header

def read_shape_gii(shape_gii):
    gii = nib.load(shape_gii)
    data = gii.darrays[0].data
    return gii, data

#### ******************************* Creat a .shape.gii file ****************************** ####

def creat_shape_gii(shape_gii_base, array_write, savepath):
    gii, _ = read_shape_gii(shape_gii_base)
    gii.darrays[0].data = array_write
    nib.save(gii, savepath)


## Creat a .shape gii file with atlas, e.g.,Kong400,Schaefer400
def creat_fs_lr32k_atlas(shape_gii_base,array_write, atlas_file, hemi, savepath):
    #cdata = np.zeros((32492,))
    f_atlas = nib.load(atlas_file)
    f_data = f_atlas.get_fdata() # 0 mean medial wall, label start from 1
    f_data = f_data[0,:]

    if hemi == 'L':
        cdata = f_data[0:int(len(f_data)/2)].copy() # L
        print('cdata.shape: %d' %(len(cdata)))
        for i, data in enumerate(array_write):
            cdata[np.where(cdata==i+1)] = data
    else:
        cdata = f_data[int(len(f_data) / 2):int(len(f_data))].copy() # R
        for i, data in enumerate(array_write):
            cdata[np.where(cdata==i+1+int(np.max(f_data)/2))] = data

    creat_shape_gii(shape_gii_base, cdata, savepath)

###################################### Corr with pvalue ###################################################
def Corr(x, y):
    r, p = pearsonr(x, y)
    rr = round(r, 3)
    pp = round(p, 3)
    return rr, pp

###################################### Join plot ###########################################################
# x,y should not be object data type, if so, convert it to np.float data type.
def jointplot_fitlinear(x, y):
    plt.figure(figsize=(20,16))
    sns.jointplot(x = x,y = y, kind = 'reg',scatter_kws={"s": 8})
#sns.regplot(x='LL_LPAC_Math_Corr',y='ReadEng_AgeAdj',data=df, color='red', scatter_kws={"s": 8}),不显示分布信息

###################################### Extract top N in a array ##############################
# Extract top N element in a array and set others to 0.
def topN_array(arr, topN):
    idx_all = set(np.arange(len(arr)))
    idx_topN = arr.argsort()[::-1][:topN]
    idx_res = idx_all - set(idx_topN)
    # keep topN, set others to zero
    idx_res = np.array(list(idx_res))
    arr_topN = arr.copy()
    arr_topN[idx_res] = 0
    return arr_topN


# def main_creat_shape_gii():
#
#     tasks = ['LANGUAGE']
#     subjs = sio.loadmat(project_path + '/Subject_taskT.mat')
#     subjects = subjs['Subject_taskT']
#     subjects = np.reshape(subjects,(len(subjects),))
#     shape_gii_base_L = project_path + '/Base_shape_gii/100206.L.thickness.32k_fs_LR.shape.gii' # for creat shape gii files
#     shape_gii_base_R = project_path + '/Base_shape_gii/100206.R.thickness.32k_fs_LR.shape.gii'  # for creat shape gii files
#
#     for subj_id in subjects:
#         df_column_name = 'task_t'
#         for task in tasks:
#             files = sorted(glob.glob(project_path + '/DataSort/Sort_Sub/' + task + '/*.h5'))
#
#             for file in files:
#
#                 Dirs = os.path.split(os.path.splitext(file)[0])
#                 kName = Dirs[1]
#                 df = rvp.read_orig_data(file, kName)
#                 ResultantFolder = project_path +'/Result/Ridge_Vert_SubBased/Statistics/task_t_gii/' + str(subj_id) +'/' + task
#                 if not os.path.exists(ResultantFolder):
#                     os.makedirs(ResultantFolder)
#                 task_t = rvp.select_data_subj(df, subj_id, 'task_t').to_numpy()[0]
#                 task_t = np.reshape(task_t, (len(task_t),))
#                 vert_index = rvp.select_data_subj(df, subj_id, 'vert_index').to_numpy()[0]
#                 cdata = np.zeros((32492,))
#                 vert_index = np.reshape(vert_index, (len(vert_index),))
#                 cdata[vert_index - 1] = task_t # compared to matlab, the vert index should minus 1
#
#                 save_name = ResultantFolder + '/' + str(subj_id) + '_' + kName + '.shape.gii'
#                 if 'LW' in file:
#                     print('LEFT SEED')
#                     shape_gii_base= shape_gii_base_L
#                 elif 'RW' in file:
#                     print('RIGHT SEED')
#                     shape_gii_base= shape_gii_base_R
#                 else:
#                     print('error!')
#
#                 creat_shape_gii(shape_gii_base, cdata, save_name)
#                 print(kName)
#
#         print(str(subj_id))














