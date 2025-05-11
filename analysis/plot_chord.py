import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.spatial.distance import squareform
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout


def top_num_ind(arr, top_num):
    return arr.argsort()[-top_num:][::-1]


def file_split(file_path):
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension


def load_model_weight_mean(filepath):
    return sio.loadmat(filepath)['mean'][0, 1:]


def creat_label_names(brain_atlas_df):

    bna_lobes = brain_atlas_df['Lobe_M']
    lobes_gyrus = brain_atlas_df['Gyrus_M']
    label_names = []
    for i in np.arange(len(bna_lobes)):
        label_names.append(bna_lobes[i].strip() + '_' + lobes_gyrus[i] + '-' + str(i))

    # for circle split
    lobe_start_index = brain_atlas_df['Lobe'].dropna().index.to_numpy()

    return label_names, lobe_start_index


def plot_chord(cmat, label_names, lobe_start_index, num_lines, lut_file, color_range, fig_out_format='svg',
               fig_out_file=None):
    node_order = label_names
    group_boundaries = lobe_start_index
    node_angles = circular_layout(label_names, node_order, start_pos=0, group_boundaries=group_boundaries, group_sep=5)

    # plot
    lut = sio.loadmat(lut_file)
    lut_9 = lut['hsv_lut_9']
    colors = lut_9[0:9, :]
    colors[2, :] = np.array([0.863, 0.957, 0.008, 1])
    colors[0, :] = np.array([0.808, 0.066, 0.348, 1])
    colors[8, :] = np.array([1, 0, 0, 1])

    node_colors = np.zeros((len(label_names), 4))
    for i, ind_i in enumerate(lobe_start_index):
        if ind_i < np.max(np.array(lobe_start_index)):
            ind_i_1 = lobe_start_index[i + 1]
            node_colors[ind_i:ind_i_1, :] = lut_9[i, :]
        else:
            node_colors[ind_i:len(label_names), :] = lut_9[i, :]

    # N_lines = 50
    colorbar_pos = (-0.5, 0.5)
    fig = plt.figure(num=None, figsize=(13, 13), facecolor='white')
    plot_connectivity_circle(cmat, label_names, n_lines=num_lines, fig=fig, node_colors=list(node_colors),
                             node_angles=node_angles, colorbar=True, vmin = color_range[0], vmax = color_range[1],
                             colorbar_pos=colorbar_pos, colormap='RdYlBu_r', subplot=111, facecolor='white',
                             textcolor='black', node_edgecolor='black', linewidth=1.0, node_linewidth=0.5)

    # save
    if fig_out_file is not None:
        fig.savefig(fig_out_file, facecolor='white', dpi=1000, format=fig_out_format)

    plt.close()


def main():
    root_dir = ""
    model_dir = "./model_weights"
    out_fig_dir = "gigasci_figures"
    bna_atlas_df = pd.read_excel(root_dir + 'BNA_subregions.xlsx')
    label_names, lobe_start_index = creat_label_names(bna_atlas_df)

    lut_path = root_dir + 'hsv_lut_9.mat'
    lobes = bna_atlas_df['Lobe_M']

    # Top 5% of the second-order weights
    thr = 0.05
    top_num = round(7503 * thr)  # 7503 is the dimension of the feature.

    data_weight_dict = {
        'HCP': {
            'male': [load_model_weight_mean('%s/first_order/HCP_L5G0_test_size00.mat' % model_dir), 'Fig_S4A'],
            'female': [load_model_weight_mean('%s/first_order/HCP_L5G1_test_size00.mat' % model_dir), 'Fig_S4B'],
            'second_order': load_model_weight_mean('%s/second_order/HCP_L5G0_vs_L5G1.mat' % model_dir)
        },
        'GSP': {
            'male': [load_model_weight_mean('%s/first_order/gsp_L5G0_test_size00.mat' % model_dir), 'Fig_S4C'],
            'female': [load_model_weight_mean('%s/first_order/gsp_L5G1_test_size00.mat' % model_dir), 'Fig_S4D'],
            'second_order': load_model_weight_mean('%s/second_order/gsp_L5G0_vs_L5G1.mat' % model_dir)
        }
    }
    model_shape = data_weight_dict['HCP']['second_order'].shape

    # create a mask by taking overlap between the 2nd order top-weights of hcp and gsp
    mask_idx = np.intersect1d(top_num_ind(np.abs(data_weight_dict["HCP"]["second_order"]), top_num),
                              top_num_ind(np.abs(data_weight_dict["GSP"]["second_order"]), top_num))

    mask = np.zeros(model_shape)
    mask[mask_idx] = 1

    mask_con_scores = squareform(mask, force='no', checks=True)

    num_lines = 1

    fig_out_fname = "%s/Fig_4A.svg" % out_fig_dir

    plot_chord(mask_con_scores, label_names, lobe_start_index, num_lines, lut_path, color_range=[None, None],
               fig_out_format="svg", fig_out_file=fig_out_fname)

    for dataset in data_weight_dict.keys():
        for group in ["male", "female"]:
            print("Processing %s %s" % (dataset, group))
            # Top 5% of the first-order weights
            top_weight_idx = top_num_ind(np.abs(data_weight_dict[dataset][group][0]), top_num)
            masked_top_weight_idx = np.intersect1d(mask_idx, top_weight_idx)
            num_lines = masked_top_weight_idx.shape[0]
            data_weight_dict[dataset][group].append(masked_top_weight_idx)
            masked_top_weights = np.zeros(model_shape)
            masked_top_weights[masked_top_weight_idx] = data_weight_dict[dataset][group][0][masked_top_weight_idx]
            masked_top_weights_con_scores = squareform(masked_top_weights, force='no', checks=True)
            fig_out_fname = "%s/%s.svg" % (out_fig_dir, data_weight_dict[dataset][group][1])
            plot_chord(masked_top_weights_con_scores, label_names, lobe_start_index, num_lines, lut_path, [-0.15, 0.15],
               fig_out_format='svg', fig_out_file=fig_out_fname)

        data_weight_dict[dataset]["overlap_masked_index"] = np.intersect1d(data_weight_dict[dataset]['male'][2],
                                                                            data_weight_dict[dataset]['female'][2])
        for group in ["male", "female"]:
            masked_top_weight_comm = np.zeros(model_shape)
            masked_top_weight_comm[data_weight_dict[dataset]["overlap_masked_index"]] = \
                data_weight_dict[dataset][group][0][data_weight_dict[dataset]["overlap_masked_index"]]
            masked_top_weight_spec_idx = np.array(list(set(data_weight_dict[dataset][group][2]) -
                                                       set(data_weight_dict[dataset]["overlap_masked_index"])))
            data_weight_dict[dataset][group].append(masked_top_weight_spec_idx)
            masked_top_weight_spec = np.zeros(model_shape)
            masked_top_weight_spec[masked_top_weight_spec_idx] = \
                data_weight_dict[dataset][group][0][masked_top_weight_spec_idx]

            out_data = {"common": masked_top_weight_comm, "specific": masked_top_weight_spec}
            for key, value in out_data.items():
                sys_mat = squareform(out_data[key], force='no', checks=True)
                df_edge = pd.DataFrame(columns=['node1_name', 'node2_name', 'edge_name', 'edge_value'])
                # sys_mat = gsp_f_com
                fname = "" + dataset + "_" + group + "_" + key
                vals = np.unique(sys_mat[sys_mat != 0])

                for k, val in enumerate(vals):
                    ind = np.where(sys_mat == val)
                    ind_1 = ind[0][0]
                    ind_2 = ind[0][1]

                    node1_name = label_names[ind_1]
                    node2_name = label_names[ind_2]
                    edge_name = node1_name + '-' + node2_name

                    df_edge.loc[k, 'node1_name'] = node1_name
                    df_edge.loc[k, 'node2_name'] = node2_name
                    df_edge.loc[k, 'edge_name'] = edge_name
                    df_edge.loc[k, 'edge_value'] = val

                # save
                out_edge_dir = os.path.join(model_dir, 'edge_vals')
                if not os.path.exists(out_edge_dir):
                    os.makedirs(out_edge_dir)
                out_edge_file = os.path.join(out_edge_dir, fname + ".csv")
                df_edge.to_csv(out_edge_file, index=False)

    #
    # # lambda=5, view the weight map
    # # HCP
    # # male
    # file_h0 = './model_weights/first_order/HCP_L5G0_test_size00.mat'
    # # female
    # file_h1 = './model_weights/first_order/HCP_L5G1_test_size00.mat'
    # ## male and female difference based model weights
    # file_h2 = './model_weights//second_order/HCP_L5G0_vs_L5G1.mat'
    #
    # wh_m = sio.loadmat(file_h0)['mean'][0, 1::]  # male, removing the constant, dimension: 7503
    # wh_f = sio.loadmat(file_h1)['mean'][0, 1::]  # female
    # wh_2 = sio.loadmat(file_h2)['mean'][0, 1::]  # second-order weights
    #
    # ## GSP
    # # male
    # file_g0 = './model_weights/first_order/gsp_L5G0_test_size00.mat'
    # # female
    # file_g1 = './model_weights/first_order/gsp_L5G1_test_size00.mat'
    #
    # ## male and female difference based model weights
    # file_g2 = './model_weights//second_order/gsp_L5G0_vs_L5G1.mat'
    #
    # wg_m = sio.loadmat(file_g0)['mean'][0, 1::]  # male
    # wg_f = sio.loadmat(file_g1)['mean'][0, 1::]  # female
    # wg_2 = sio.loadmat(file_g2)['mean'][0, 1::]  # second-order weights
    #
    # # Top 5% of the second-order weights
    # thr = 0.05
    # top_num = round(7503 * thr)  # 7503 is the dimension of the feature.
    # # M and F
    # wh_thr2 = top_num_ind(np.abs(wh_2), top_num)  # 2阶
    # wg_thr2 = top_num_ind(np.abs(wg_2), top_num)  # 2阶
    # ind_olp = np.intersect1d(wh_thr2, wg_thr2)  # overlap hcp and gsp
    #
    # whg_mask = np.zeros(wh_2.shape)
    # whg_mask[ind_olp] = 1
    #
    # mask_con_scores = squareform(whg_mask, force='no', checks=True)
    #
    # num_lines = 1
    #
    # fig_out_fname = "%s/Fig_4A.svg" % out_fig_dir
    #
    # plot_chord(mask_con_scores, label_names, lobe_start_index, num_lines, lut_path, color_range = [None, None],
    #            fig_out_format="svg", fig_out_file=fig_out_fname)
    #
    # # HCP-m
    # wh_m_th1 = top_num_ind(np.abs(wh_m), top_num)  # absolute, first-order
    # wh_ind_olp_m = np.intersect1d(ind_olp, wh_m_th1)  # overlap index with the second-order mask
    #
    # # HCP-f
    # wh_f_th1 = top_num_ind(np.abs(wh_f), top_num)
    # wh_ind_olp_f = np.intersect1d(ind_olp, wh_f_th1)
    #
    # wh_m_olp = np.zeros((len(wh_m),))
    # wh_m_olp[wh_ind_olp_m] = wh_m[wh_ind_olp_m]  # the oringal first-order masked model weights.
    #
    # wh_f_olp = np.zeros((len(wh_f),))
    # wh_f_olp[wh_ind_olp_f] = wh_f[wh_ind_olp_f]
    #
    # fig_out_fname = "%s/Fig_S4A.svg" % out_fig_dir
    # plot_chord(wh_m_olp, label_names, lobe_start_index, num_lines, lut_path, [-0.15, 0.15],
    #            fig_out_format='svg', fig_out_file=fig_out_fname)
    #
    # fig_out_fname = "%s/Fig_S4B.svg" % out_fig_dir
    # plot_chord(wh_f_olp, label_names, lobe_start_index, num_lines, lut_path, [-0.15, 0.15],
    #            fig_out_format='svg', fig_out_file=fig_out_fname)
    #
    # # GSP-m
    # wg_m_th1 = top_num_ind(np.abs(wg_m), top_num)  # absolute, first-order
    # wg_ind_olp_m = np.intersect1d(ind_olp, wg_m_th1)  # overlap index with the second-order mask
    #
    # # GSP-f
    # wg_f_th1 = top_num_ind(np.abs(wg_f), top_num)
    # wg_ind_olp_f = np.intersect1d(ind_olp, wg_f_th1)
    #
    # wg_m_olp = np.zeros((len(wg_m),))
    # wg_m_olp[wg_ind_olp_m] = wg_m[wg_ind_olp_m]  # the oringal first-order masked model weights.
    #
    # wg_f_olp = np.zeros((len(wg_f),))
    # wg_f_olp[wg_ind_olp_f] = wg_f[wg_ind_olp_f]
    #
    # fig_out_fname = "%s/Fig_S4C.svg" % out_fig_dir
    # plot_chord(wg_m_olp, label_names, lobe_start_index, num_lines, lut_path, [-0.15, 0.15],
    #            fig_out_format='svg', fig_out_file=fig_out_fname)
    #
    # fig_out_fname = "%s/Fig_S4D.svg" % out_fig_dir
    # plot_chord(wg_f_olp, label_names, lobe_start_index, num_lines, lut_path, [-0.15, 0.15],
    #            fig_out_format='svg', fig_out_file=fig_out_fname )
    #
    # wh_ind_mf = np.intersect1d(wh_ind_olp_m, wh_ind_olp_f)  # hcp M and F
    # wg_ind_mf = np.intersect1d(wg_ind_olp_m, wg_ind_olp_f)  # gsp M and F
    #
    # ## HCP common MF
    # # m
    # wh_m_olpmf = np.zeros((len(wh_m),))
    # wh_m_olpmf[wh_ind_mf] = wh_m[wh_ind_mf]
    #
    # # f
    # wh_f_olpmf = np.zeros((len(wh_f),))
    # wh_f_olpmf[wh_ind_mf] = wh_f[wh_ind_mf]
    #
    # # HCP MF spec
    # # # m
    # wh_m_spec_ind = set(wh_ind_olp_m) - set(wh_ind_mf)
    # wh_m_spec_ind = np.array(list(wh_m_spec_ind))
    #
    # wh_m_spec = np.zeros((len(wh_m),))
    # wh_m_spec[wh_m_spec_ind] = wh_m[wh_m_spec_ind]
    #
    # # f
    # wh_f_spec_ind = set(wh_ind_olp_f) - set(wh_ind_mf)
    # wh_f_spec_ind = np.array(list(wh_f_spec_ind))
    #
    # wh_f_spec = np.zeros((len(wh_f),))
    # wh_f_spec[wh_f_spec_ind] = wh_f[wh_f_spec_ind]
    #
    # sys_wh_m_olpmf = squareform(wh_m_olpmf, force='no', checks=True)
    # sys_wh_f_olpmf = squareform(wh_f_olpmf, force='no', checks=True)
    # sys_wh_m_spec = squareform(wh_m_spec, force='no', checks=True)
    # sys_wh_f_spec = squareform(wh_f_spec, force='no', checks=True)
    #
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_m_com.mat',
    #             {'sys_wh_m_olpmf': sys_wh_m_olpmf})
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_f_com.mat',
    #             {'sys_wh_f_olpmf': sys_wh_f_olpmf})
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_m_spec.mat',
    #             {'sys_wh_m_spec': sys_wh_m_spec})
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_f_spec.mat',
    #             {'sys_wh_f_spec': sys_wh_f_spec})
    #
    # ## GSP common MF
    # # m
    # wg_m_olpmf = np.zeros((len(wg_m),))
    # wg_m_olpmf[wg_ind_mf] = wg_m[wg_ind_mf]
    #
    # # f
    # wg_f_olpmf = np.zeros((len(wg_f),))
    # wg_f_olpmf[wg_ind_mf] = wg_f[wg_ind_mf]
    #
    # ## GSP MF spec
    # # m
    # wg_m_spec_ind = set(wg_ind_olp_m) - set(wg_ind_mf)
    # wg_m_spec_ind = np.array(list(wg_m_spec_ind))
    #
    # wg_m_spec = np.zeros((len(wg_m),))
    # wg_m_spec[wg_m_spec_ind] = wg_m[wg_m_spec_ind]
    #
    # # f
    # wg_f_spec_ind = set(wg_ind_olp_f) - set(wg_ind_mf)
    # wg_f_spec_ind = np.array(list(wg_f_spec_ind))
    #
    # wg_f_spec = np.zeros((len(wg_f),))
    # wg_f_spec[wg_f_spec_ind] = wg_f[wg_f_spec_ind]
    #
    # sys_wg_m_olpmf = squareform(wg_m_olpmf, force='no', checks=True)
    # sys_wg_f_olpmf = squareform(wg_f_olpmf, force='no', checks=True)
    # sys_wg_m_spec = squareform(wg_m_spec, force='no', checks=True)
    # sys_wg_f_spec = squareform(wg_f_spec, force='no', checks=True)
    #
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_m_com.mat',
    #             {'sys_wg_m_olpmf': sys_wg_m_olpmf})
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_f_com.mat',
    #             {'sys_wg_f_olpmf': sys_wg_f_olpmf})
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_m_spec.mat',
    #             {'sys_wg_m_spec': sys_wg_m_spec})
    # sio.savemat('/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_f_spec.mat',
    #             {'sys_wg_f_spec': sys_wg_f_spec})
    #
    # # hcp
    # hcp_m_com = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_m_com.mat')
    # hcp_m_com = hcp_m_com['sys_wh_m_olpmf']
    #
    # hcp_f_com = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_f_com.mat')
    # hcp_f_com = hcp_f_com['sys_wh_f_olpmf']
    #
    # hcp_m_spec = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_m_spec.mat')
    # hcp_m_spec = hcp_m_spec['sys_wh_m_spec']
    #
    # hcp_f_spec = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/hcp_f_spec.mat')
    # hcp_f_spec = hcp_f_spec['sys_wh_f_spec']
    #
    # # gsp
    #
    # gsp_m_com = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_m_com.mat')
    # gsp_m_com = gsp_m_com['sys_wg_m_olpmf']
    #
    # gsp_f_com = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_f_com.mat')
    # gsp_f_com = gsp_f_com['sys_wg_f_olpmf']
    #
    # gsp_m_spec = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_m_spec.mat')
    # gsp_m_spec = gsp_m_spec['sys_wg_m_spec']
    #
    # gsp_f_spec = sio.loadmat(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/BNA/gsp_f_spec.mat')
    # gsp_f_spec = gsp_f_spec['sys_wg_f_spec']
    #
    # df_edge = pd.DataFrame(columns=['node1_name', 'node2_name', 'edge_name', 'edge_value'])
    # sys_mat = gsp_f_com
    # fname = 'gsp_f_com'
    # vals = np.unique(sys_mat[sys_mat != 0])
    #
    # for k, val in enumerate(vals):
    #     ind = np.where(sys_mat == val)
    #     ind_1 = ind[0][0]
    #     ind_2 = ind[0][1]
    #
    #     node1_name = label_names[ind_1]
    #     node2_name = label_names[ind_2]
    #     edge_name = node1_name + '-' + node2_name
    #
    #     df_edge.loc[k, 'node1_name'] = node1_name
    #     df_edge.loc[k, 'node2_name'] = node2_name
    #     df_edge.loc[k, 'edge_name'] = edge_name
    #     df_edge.loc[k, 'edge_value'] = val
    #
    # # save
    #
    # df_edge.to_csv(
    #     '/Users/fiona/Junhao/Project/BNU_SHFE/Final_Version/Result/Figs_0717/brain_net/edge_vals/' + fname + '.csv',
    #     index=False)


if __name__ == '__main__':
    main()
