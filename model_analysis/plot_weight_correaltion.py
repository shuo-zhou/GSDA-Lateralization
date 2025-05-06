# import os
#
# import matplotlib.pylab as plt
# import numpy as np
# # import pandas as pd
# # import seaborn as sns
# # from plot_classification_results import savefig
# from scipy.io import loadmat
#
#
# def load_weight_plot_corr(dataset, base_dir, test_size="00", fontsize=14):
#     control_file = os.path.join(
#         base_dir, "%s_L0_group_mixG0_test_size%s.mat" % (dataset, test_size)
#     )
#     control_weights = loadmat(control_file)["mean"][0]
#
#     n_control_weights = control_weights.shape[0]
#
#     lambdas = [0, 1, 2, 5, 8, 10]
#     corrs = {"mean": [], "sd": []}
#
#     for lambda_ in lambdas:
#         for group in [0, 1]:
#             weight_file = os.path.join(
#                 base_dir,
#                 "%s_L%dG%d_test_size%s.mat" % (dataset, int(lambda_), group, test_size),
#             )
#             weights = loadmat(weight_file)["mean"][0]
#             # n_weights = weights.shape[0]
#             corr_matrix = np.corrcoef(control_weights, weights)[
#                 n_control_weights:, :n_control_weights
#             ]
#             corrs["mean"].append(np.mean(corr_matrix))
#             corrs["std"].append(np.std(corr_matrix))
#
#     plt.rcParams.update({"font.size": fontsize})
#     fig, ax = plt.subplots(2, 1, sharex=True)
#     fig.set_size_inches(4, 8.5)
#     # .plot(x, y1)
#     # .plot(x, y2)
#
#     # plt.figure(figsize=(4, 4))
#     ax[0].plot(lambdas, corrs["mean"][::2], "-", c="steelblue", label="Male-specific")
#     ax[0].fill_between(
#         lambdas,
#         np.asarray(corrs["mean"][::2]) - np.asarray(corrs["sd"][::2]),
#         np.asarray(corrs["mean"][::2]) + np.asarray(corrs["sd"][::2]),
#         color="steelblue",
#         alpha=0.2,
#     )
#     ax[0].set_ylabel("Correlation")
#     ax[0].legend(loc="upper right")
#
#     ax[1].plot(
#         lambdas, corrs["mean"][1::2], "--", color="firebrick", label="Female-specific"
#     )
#     ax[1].fill_between(
#         lambdas,
#         np.asarray(corrs["mean"][1::2]) - np.asarray(corrs["std"][1::2]),
#         np.asarray(corrs["mean"][1::2]) + np.asarray(corrs["std"][1::2]),
#         color="firebrick",
#         alpha=0.2,
#     )
#
#     ax[1].set_ylabel("Correlation")
#     ax[1].legend(loc="upper right")
#     plt.rcParams["text.usetex"] = True
#     plt.xlabel(r"$\lambda$", fontsize=fontsize)
#     plt.rcParams["text.usetex"] = False
#
#     plt.savefig("figures/%s_corr.svg" % dataset, format="svg", bbox_inches="tight")
#     # plt.savefig('figures/%s_corr.pdf' % dataset, format='pdf', bbox_inches='tight')
#     # plt.savefig('figures/%s_corr.png' % dataset, format='png', bbox_inches='tight')
#     plt.show()
#
#
# def main():
#     datasets = ["HCP", "gsp"]
#     base_dir = "model_weights/first_order"
#     for dataset in datasets:
#         load_weight_plot_corr(dataset, base_dir)
#
#     # dataset = "gsp"
#     # base_dir = "/media/shuo/MyDrive/data/brain/brain_networks/gsp/Models"
#     # sessions = [""]
#     # seed_ = 2023
#     #
#     # load_weight_plot_corr(dataset, base_dir, sessions, seed_)
#     #
#     # dataset = "HCP"
#     # base_dir = "/media/shuo/MyDrive/data/HCP/BNA/Models"
#     # sessions = ["REST1_", "REST2_"]
#     # seed_ = 2022
#     #
#     # load_weight_plot_corr(dataset, base_dir, sessions, seed_)
#
#     # weights_dirs = {
#     #     "Multivariate Control HCP": "./first-order/HCP/HCP_L0Gmix.mat",
#     #     "GSDA $\lambda=0$ Male HCP": "./first-order/HCP/HCP_L0G0.mat",
#     #     "GSDA $\lambda=0$ Female HCP": "./first-order/HCP/HCP_L0G1.mat",
#     #     "Multivariate Control GSP": "./first-order/GSP/GSP_L0_mix_genderGmix.mat",
#     #     "GSDA $\lambda=0$ Male GSP": "./first-order/GSP/GSP_L0G0.mat",
#     #     "GSDA $\lambda=0$ Female GSP": "./first-order/GSP/GSP_L0G1.mat",
#     #     "GSDA $\lambda=5$ Male HCP": "./first-order/HCP/HCP_L5G0.mat",
#     #     "GSDA $\lambda=5$ Female HCP": "./first-order/HCP/HCP_L5G1.mat",
#     #     "GSDA $\lambda=5$ Male GSP": "./first-order/GSP/GSP_L5G0.mat",
#     #     "GSDA $\lambda=5$ Female GSP": "./first-order/GSP/GSP_L5G1.mat",
#     # }
#     #
#     # weights = dict()
#     #
#     # for key in weights_dirs:
#     #     # print(key)
#     #     weights[key] = loadmat(weights_dirs[key])["mean"][0][1:]
#     #
#     # weight_df = pd.DataFrame(weights)
#     #
#     # HCP_tval = pd.read_csv(
#     #     "../post_analysis/univariate/REST1_univariate_L_vs_R_gender_tvalue_uncorrected.csv"
#     # )
#     # HCP_tval = HCP_tval.rename(
#     #     columns={
#     #         "M_F": "Univariate Control HCP ",
#     #         "M": "Univariate Male HCP",
#     #         "F": "Univariate Female HCP",
#     #     }
#     # )
#     # GSP_tval = pd.read_csv(
#     #     "../post_analysis/univariate/GSP_univariate_L_vs_R_gender_tvalue_uncorrected.csv"
#     # )
#     # GSP_tval = GSP_tval.rename(
#     #     columns={
#     #         "M_F": "Univariate Control GSP ",
#     #         "M": "Univariate Male GSP",
#     #         "F": "Univariate Female GSP",
#     #     }
#     # )
#     #
#     # weight_df = pd.concat((HCP_tval, GSP_tval, weight_df), axis=1)
#     #
#     # fontsize = 14
#     # corr = weight_df.corr()
#     #
#     # # Generate a mask for the upper triangle
#     # mask = np.triu(np.ones_like(corr, dtype=bool))
#     #
#     # # Set up the matplotlib figure
#     # f, ax = plt.subplots(figsize=(11, 9))
#     # plt.rcParams.update({"font.size": fontsize})
#     # # Generate a custom diverging colormap
#     # cmap = sns.diverging_palette(230, 20, as_cmap=True)
#     # plt.rcParams["text.usetex"] = True
#     # # Draw the heatmap with the mask and correct aspect ratio
#     # sns.heatmap(
#     #     corr,
#     #     mask=mask,
#     #     cmap=cmap,
#     #     vmin=0,
#     #     vmax=1,
#     #     center=0.5,
#     #     annot=True,
#     #     square=True,
#     #     linewidths=0.5,
#     #     cbar_kws={"shrink": 0.5},
#     # )
#     # # cbar_kws={"shrink": .5, "use_gridspec": False, "location": "top"})  #, annot_kws={"rotation": 45})
#     # plt.rcParams["text.usetex"] = False
#     # # plt.tick_params(axis='x', labelrotation=45)
#     # ax.set_xticklabels(list(weight_df.columns.values), rotation=45, ha="right")
#     # plt.savefig("figures/corr_annot.svg", format="svg", bbox_inches="tight")
#     # # plt.savefig('figures/corr_annot.pdf', format='pdf', bbox_inches='tight')
#     # # plt.savefig('figures/corr_annot.png', format='png', bbox_inches='tight')
#     # plt.show()
#
#
# if __name__ == "__main__":
#     main()
