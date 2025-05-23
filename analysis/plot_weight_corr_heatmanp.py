import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat


def load_weights_dict(base_dir, dataset, group_label, lambdas, test_size="00"):
    weights = {}
    for lam in lambdas:
        key = r"$\lambda=%d$" % (int(lam))
        fname = f"{dataset}_L{int(lam)}G{group_label}_test_size{test_size}.mat"
        fpath = os.path.join(base_dir, "first_order", fname)
        weights[key] = loadmat(fpath)["mean"][0][1:]
    return weights


def plot_corr_heatmap(df, outfile, fontsize=14):
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), 1)

    plt.figure(figsize=(11, 9))
    sns.set_context("notebook", font_scale=1.2)
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    ax = sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        annot=True,
        annot_kws={"fontsize": fontsize},
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
    )
    ax.set_xticklabels(df.columns, rotation=45, ha="right", fontsize=fontsize)
    ax.set_yticklabels(df.columns, rotation=0, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outfile + ".svg", format="svg")
    plt.savefig(outfile + ".pdf", format="pdf")
    plt.show()


def main():
    base_dir = "model_weights"
    out_fig_dir = "gigasci_figures"
    # fontsize = 14
    weights_dirs = {
        "Multivariate Control HCP": "%s/first_order/HCP_L0_group_mixG0_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=0$ Male HCP": "%s/first_order/HCP_L0G0_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=0$ Female HCP": "%s/first_order/HCP_L0G1_test_size00.mat"
        % base_dir,
        "Multivariate Control GSP": "%s/first_order/gsp_L0_group_mixG0_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=0$ Male GSP": "%s/first_order/gsp_L0G0_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=0$ Female GSP": "%s/first_order/gsp_L0G1_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=5$ Male HCP": "%s/first_order/HCP_L5G0_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=5$ Female HCP": "%s/first_order/HCP_L5G1_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=5$ Male GSP": "%s/first_order/gsp_L5G0_test_size00.mat"
        % base_dir,
        r"GSDA $\lambda=5$ Female GSP": "%s/first_order/gsp_L5G1_test_size00.mat"
        % base_dir,
    }

    weights = dict()

    for key in weights_dirs:
        # print(key)
        weights[key] = loadmat(weights_dirs[key])["mean"][0][1:]

    weight_df = pd.DataFrame(weights)

    hcp_tval = pd.read_csv(
        "%s/univariate/REST1_univariate_L_vs_R_gender_tvalue_uncorrected.csv" % base_dir
    )
    hcp_tval = hcp_tval.rename(
        columns={
            "M_F": "Univariate Control HCP ",
            "M": "Univariate Male HCP",
            "F": "Univariate Female HCP",
        }
    )
    gsp_tval = pd.read_csv(
        "%s/univariate/GSP_univariate_L_vs_R_gender_tvalue_uncorrected.csv" % base_dir
    )
    gsp_tval = gsp_tval.rename(
        columns={
            "M_F": "Univariate Control GSP ",
            "M": "Univariate Male GSP",
            "F": "Univariate Female GSP",
        }
    )

    weight_df = pd.concat((hcp_tval, gsp_tval, weight_df), axis=1)

    plot_corr_heatmap(weight_df, "%s/Fig_3A" % out_fig_dir, fontsize=12)

    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]

    outfile_dict = {
        "HCP": {0: "Fig_3C", 1: "Fig_3D"},
        "gsp": {0: "Fig_S3C", 1: "Fig_S3D"},
    }

    for dataset in outfile_dict.keys():
        for group_label in outfile_dict[dataset].keys():
            weights = load_weights_dict(base_dir, dataset, group_label, lambdas)
            df = pd.DataFrame(weights)
            plot_corr_heatmap(
                df,
                os.path.join(out_fig_dir, outfile_dict[dataset][group_label]),
                fontsize=14,
            )


if __name__ == "__main__":
    main()
