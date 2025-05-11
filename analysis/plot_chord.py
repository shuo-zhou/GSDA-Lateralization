import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
from mne.viz import circular_layout
from mne_connectivity.viz import plot_connectivity_circle
from scipy.spatial.distance import squareform


def top_num_ind(arr, top_num):
    return arr.argsort()[-top_num:][::-1]


def file_split(file_path):
    filepath, tempfilename = os.path.split(file_path)
    filename, extension = os.path.splitext(tempfilename)
    return filepath, filename, extension


def load_model_weight_mean(filepath):
    return sio.loadmat(filepath)["mean"][0, 1:]


def creat_label_names(brain_atlas_df):
    bna_lobes = brain_atlas_df["Lobe_M"]
    lobes_gyrus = brain_atlas_df["Gyrus_M"]
    label_names = []
    for i in np.arange(len(bna_lobes)):
        label_names.append(bna_lobes[i].strip() + "_" + lobes_gyrus[i] + "-" + str(i))

    # for circle split
    lobe_start_index = brain_atlas_df["Lobe"].dropna().index.to_numpy()

    return label_names, lobe_start_index


def plot_chord(
    cmat,
    label_names,
    lobe_start_index,
    num_lines,
    lut_file,
    color_range,
    fig_out_format="svg",
    fig_out_file=None,
):
    node_order = label_names
    group_boundaries = lobe_start_index
    node_angles = circular_layout(
        label_names,
        node_order,
        start_pos=0,
        group_boundaries=group_boundaries,
        group_sep=5,
    )

    # plot
    lut = sio.loadmat(lut_file)
    lut_9 = lut["hsv_lut_9"]
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
            node_colors[ind_i : len(label_names), :] = lut_9[i, :]

    # N_lines = 50
    colorbar_pos = (-0.5, 0.5)
    fig = plt.figure(num=None, figsize=(13, 13), facecolor="white")
    plot_connectivity_circle(
        cmat,
        label_names,
        n_lines=num_lines,
        fig=fig,
        node_colors=list(node_colors),
        node_angles=node_angles,
        colorbar=True,
        vmin=color_range[0],
        vmax=color_range[1],
        colorbar_pos=colorbar_pos,
        colormap="RdYlBu_r",
        subplot=111,
        facecolor="white",
        textcolor="black",
        node_edgecolor="black",
        linewidth=1.0,
        node_linewidth=0.5,
    )

    # save
    if fig_out_file is not None:
        fig.savefig(fig_out_file, facecolor="white", dpi=1000, format=fig_out_format)

    plt.close()


def main():
    root_dir = ""
    model_dir = "./model_weights"
    out_fig_dir = "gigasci_figures"
    bna_atlas_df = pd.read_excel(root_dir + "BNA_subregions.xlsx")
    label_names, lobe_start_index = creat_label_names(bna_atlas_df)

    lut_path = root_dir + "hsv_lut_9.mat"
    # lobes = bna_atlas_df["Lobe_M"]

    # Top 5% of the second-order weights
    thr = 0.05
    top_num = round(7503 * thr)  # 7503 is the dimension of the feature.

    data_weight_dict = {
        "HCP": {
            "male": [
                load_model_weight_mean(
                    "%s/first_order/HCP_L5G0_test_size00.mat" % model_dir
                ),
                "Fig_S4A",
            ],
            "female": [
                load_model_weight_mean(
                    "%s/first_order/HCP_L5G1_test_size00.mat" % model_dir
                ),
                "Fig_S4B",
            ],
            "second_order": load_model_weight_mean(
                "%s/second_order/HCP_L5G0_vs_L5G1.mat" % model_dir
            ),
        },
        "GSP": {
            "male": [
                load_model_weight_mean(
                    "%s/first_order/gsp_L5G0_test_size00.mat" % model_dir
                ),
                "Fig_S4C",
            ],
            "female": [
                load_model_weight_mean(
                    "%s/first_order/gsp_L5G1_test_size00.mat" % model_dir
                ),
                "Fig_S4D",
            ],
            "second_order": load_model_weight_mean(
                "%s/second_order/gsp_L5G0_vs_L5G1.mat" % model_dir
            ),
        },
    }
    model_shape = data_weight_dict["HCP"]["second_order"].shape

    # create a mask by taking overlap between the 2nd order top-weights of hcp and gsp
    mask_idx = np.intersect1d(
        top_num_ind(np.abs(data_weight_dict["HCP"]["second_order"]), top_num),
        top_num_ind(np.abs(data_weight_dict["GSP"]["second_order"]), top_num),
    )

    mask = np.zeros(model_shape)
    mask[mask_idx] = 1

    mask_con_scores = squareform(mask, force="no", checks=True)

    num_lines = 1

    fig_out_fname = "%s/Fig_4A.svg" % out_fig_dir

    plot_chord(
        mask_con_scores,
        label_names,
        lobe_start_index,
        num_lines,
        lut_path,
        color_range=[None, None],
        fig_out_format="svg",
        fig_out_file=fig_out_fname,
    )

    for dataset in data_weight_dict.keys():
        for group in ["male", "female"]:
            print("Processing %s %s" % (dataset, group))
            # Top 5% of the first-order weights
            top_weight_idx = top_num_ind(
                np.abs(data_weight_dict[dataset][group][0]), top_num
            )
            masked_top_weight_idx = np.intersect1d(mask_idx, top_weight_idx)
            num_lines = masked_top_weight_idx.shape[0]
            data_weight_dict[dataset][group].append(masked_top_weight_idx)
            masked_top_weights = np.zeros(model_shape)
            masked_top_weights[masked_top_weight_idx] = data_weight_dict[dataset][
                group
            ][0][masked_top_weight_idx]
            masked_top_weights_con_scores = squareform(
                masked_top_weights, force="no", checks=True
            )
            fig_out_fname = "%s/%s.svg" % (
                out_fig_dir,
                data_weight_dict[dataset][group][1],
            )
            plot_chord(
                masked_top_weights_con_scores,
                label_names,
                lobe_start_index,
                num_lines,
                lut_path,
                [-0.15, 0.15],
                fig_out_format="svg",
                fig_out_file=fig_out_fname,
            )

        data_weight_dict[dataset]["overlap_masked_index"] = np.intersect1d(
            data_weight_dict[dataset]["male"][2], data_weight_dict[dataset]["female"][2]
        )
        for group in ["male", "female"]:
            masked_top_weight_comm = np.zeros(model_shape)
            masked_top_weight_comm[
                data_weight_dict[dataset]["overlap_masked_index"]
            ] = data_weight_dict[dataset][group][0][
                data_weight_dict[dataset]["overlap_masked_index"]
            ]
            masked_top_weight_spec_idx = np.array(
                list(
                    set(data_weight_dict[dataset][group][2])
                    - set(data_weight_dict[dataset]["overlap_masked_index"])
                )
            )
            data_weight_dict[dataset][group].append(masked_top_weight_spec_idx)
            masked_top_weight_spec = np.zeros(model_shape)
            masked_top_weight_spec[masked_top_weight_spec_idx] = data_weight_dict[
                dataset
            ][group][0][masked_top_weight_spec_idx]

            out_data = {
                "common": masked_top_weight_comm,
                "specific": masked_top_weight_spec,
            }
            for key, value in out_data.items():
                sys_mat = squareform(out_data[key], force="no", checks=True)
                df_edge = pd.DataFrame(
                    columns=["node1_name", "node2_name", "edge_name", "edge_value"]
                )
                # sys_mat = gsp_f_com
                fname = "" + dataset + "_" + group + "_" + key
                vals = np.unique(sys_mat[sys_mat != 0])

                for k, val in enumerate(vals):
                    ind = np.where(sys_mat == val)
                    ind_1 = ind[0][0]
                    ind_2 = ind[0][1]

                    node1_name = label_names[ind_1]
                    node2_name = label_names[ind_2]
                    edge_name = node1_name + "-" + node2_name

                    df_edge.loc[k, "node1_name"] = node1_name
                    df_edge.loc[k, "node2_name"] = node2_name
                    df_edge.loc[k, "edge_name"] = edge_name
                    df_edge.loc[k, "edge_value"] = val

                # save
                out_edge_dir = os.path.join(model_dir, "edge_vals")
                if not os.path.exists(out_edge_dir):
                    os.makedirs(out_edge_dir)
                out_edge_file = os.path.join(out_edge_dir, fname + ".csv")
                df_edge.to_csv(out_edge_file, index=False)


if __name__ == "__main__":
    main()
