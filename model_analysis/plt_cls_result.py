import matplotlib.pylab as plt
from scipy.io import loadmat

import os
import sys
sys.path.insert(0, '..')

from utils.io_ import load_result, reformat_results
# from plot import plot_accuracy, plot_gsi, load_weight_plot_corr, load_coef_plot_corr
from utils import plot


def label_gender(df):
    df = df.copy()
    df["train_gender"] = df["train_gender"].astype(str)
    df.loc[df["train_gender"] == "0", "train_gender"] = "Male"
    df.loc[df["train_gender"] == "1", "train_gender"] = "Female"
    return df

def plot_hcp(res_df_all, test_size, out_dir="figures"):
    res_df_all["GSI_is"] = 2 * (res_df_all["acc_ic_is"] * (res_df_all["acc_ic_is"] - res_df_all["acc_oc_is"]))
    res_df_all["GSI_os"] = 2 * (res_df_all["acc_ic_os"] * (res_df_all["acc_ic_os"] - res_df_all["acc_oc_os"]))
    
    res_df_is = reformat_results(res_df_all, ["acc_ic_is", "acc_oc_is"])
    res_df_os = reformat_results(res_df_all, ["acc_ic_os", "acc_oc_os"])

    res_df_all = label_gender(res_df_all)

    test_size_str = str(test_size).replace(".", "-")
    outfig_format = ["svg", "pdf"]

    plot.plot_accuracy(res_df_is, outfile=f"{out_dir}/HCP_in_session_test{test_size_str}",
                       outfig_format=outfig_format)
    plot.plot_accuracy(res_df_os, outfile=f"{out_dir}/HCP_out_session_test{test_size_str}",
                       outfig_format=outfig_format)

    plot.plot_gsi(res_df_all, x="lambda", y="GSI_is", hue="train_gender",
                  outfile=f"{out_dir}/GSI_box_HCP_in_session_test", outfig_format=outfig_format)
    plot.plot_gsi(res_df_all, x="lambda", y="GSI_os", hue="train_gender",
                  outfile=f"{out_dir}/GSI_box_HCP_out_session_test", outfig_format=outfig_format)

    if test_size == 0.2:
        res_df_test_sub = reformat_results(res_df_all, ["acc_tgt_test_sub", "acc_nt_test_sub"])
        plot.plot_accuracy(res_df_test_sub,
                           outfile=f"{out_dir}/HCP_test_sub_test{test_size_str}", outfig_format=outfig_format)
        res_df_all["GSI_test_sub"] = 2 * (
                    res_df_all["acc_tgt_test_sub"] * (res_df_all["acc_tgt_test_sub"] - res_df_all["acc_nt_test_sub"]))
        plot.plot_gsi(res_df_all, x="lambda", y="GSI_test_sub", hue="train_gender",
                      outfile=f"{out_dir}/GSI_box_HCP_test_sub_test{test_size_str}", outfig_format=outfig_format)

def plot_gsp(res_df_all, out_dir="figures"):
    res_df_all["gap"] = res_df_all["acc_ic"] - res_df_all["acc_oc"]
    res_df_all["GSI"] = 2 * (res_df_all["acc_ic"] * (res_df_all["acc_ic"] - res_df_all["acc_oc"]))

    res_df_reformat = reformat_results(res_df_all, ["acc_ic", "acc_oc"])

    res_df_all = label_gender(res_df_all)

    plot.plot_accuracy(res_df_reformat, outfile=f"{out_dir}/GSP_test", outfig_format=["svg", "pdf"])
    plot.plot_gsi(res_df_all, x="lambda", y="GSI", hue="train_gender", outfile=f"{out_dir}/GSI_box_GSP_test",
                  outfig_format = ["svg", "pdf"])


def main():
    model_root_dir = "cls_results"
    seed_start = 2023
    datasets = {"gsp": [0.0], "HCP": [0.0, 0.2]}
    out_dir = "cls_figures"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]
    
    for dataset in datasets.keys():
        result_dir = f"{model_root_dir}/{dataset}/"
        for test_size in datasets[dataset]:
            res_df_all = load_result(dataset=dataset, root_dir=result_dir, lambdas=lambdas, seed_start=seed_start, 
                                     test_size=test_size)

            if dataset == "HCP":
                plot_hcp(res_df_all, test_size, out_dir=out_dir)
            elif dataset == "gsp":
                plot_gsp(res_df_all, out_dir=out_dir)


if __name__ == "__main__":
    main()
