import os

import matplotlib.pylab as plt
import numpy as np

# from plot_classification_results import savefig


def load_weight_plot_corr(
    dataset,
    base_dir,
    test_size="00",
    n_rows=2,
    n_cols=1,
    width=4,
    height=8.5,
    fontsize=14,
    outdir="figures",
):
    control_file = os.path.join(
        base_dir, "%s_L0_group_mixG0_test_size%s.npz" % (dataset, test_size)
    )
    control_weights = np.load(control_file)["arr_0"]

    n_controls = control_weights.shape[0]

    lambdas = [0, 1, 2, 5, 8, 10]
    corrs = {"mean": [], "std": []}

    for lambda_ in lambdas:
        for group in [0, 1]:
            weight_file = os.path.join(
                base_dir,
                "%s_L%dG%d_test_size%s.npz" % (dataset, int(lambda_), group, test_size),
            )
            weights = np.load(weight_file)["arr_0"]
            n_models = weights.shape[0]
            # Take lower square of the correlation matrix for the correlations between control and gsda model weights
            corr_matrix = np.corrcoef(control_weights, weights)[n_controls:, :n_models]
            corrs["mean"].append(np.mean(corr_matrix))
            corrs["std"].append(np.std(corr_matrix))

    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(n_rows, n_cols, sharex=True)
    fig.set_size_inches(width, height)

    ax[0].plot(lambdas, corrs["mean"][::2], "-", c="steelblue", label="Male-specific")
    ax[0].fill_between(
        lambdas,
        np.asarray(corrs["mean"][::2]) - np.asarray(corrs["std"][::2]),
        np.asarray(corrs["mean"][::2]) + np.asarray(corrs["std"][::2]),
        color="steelblue",
        alpha=0.2,
    )
    ax[0].set_ylabel("Correlation")
    ax[0].legend(loc="upper right")

    ax[1].plot(
        lambdas, corrs["mean"][1::2], "--", color="firebrick", label="Female-specific"
    )
    ax[1].fill_between(
        lambdas,
        np.asarray(corrs["mean"][1::2]) - np.asarray(corrs["std"][1::2]),
        np.asarray(corrs["mean"][1::2]) + np.asarray(corrs["std"][1::2]),
        color="firebrick",
        alpha=0.2,
    )

    ax[1].set_ylabel("Correlation")
    ax[1].legend(loc="upper right")
    plt.rcParams["text.usetex"] = True
    plt.xlabel(r"$\lambda$", fontsize=fontsize)
    plt.rcParams["text.usetex"] = False

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if dataset == "HCP":
        out_fig_path = "%s/Fig_3B" % outdir
    elif dataset == "gsp":
        out_fig_path = "%s/Fig_S3AS3B" % outdir
    else:
        raise ValueError("dataset must be HCP or gsp")

    plt.savefig("%s.svg" % out_fig_path, format="svg", bbox_inches="tight")
    plt.savefig("%s.pdf" % out_fig_path, format="pdf", bbox_inches="tight")
    # plt.savefig('figures/%s_corr.png' % dataset, format='png', bbox_inches='tight')
    plt.show()


def main():
    datasets = ["HCP", "gsp"]
    base_dir = "model_weights/first_order"
    out_fig_dir = "gigasci_figures"
    for dataset in datasets:
        load_weight_plot_corr(dataset, base_dir, outdir=out_fig_dir)


if __name__ == "__main__":
    main()
