import os

import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns

PLOT_FORMATS = ["svg", "pdf"]


def savefig(fig, outfile, outfig_format):
    if isinstance(outfig_format, list):
        for fmt in outfig_format:
            fig.savefig("%s.%s" % (outfile, fmt), format=fmt, bbox_inches="tight")
    else:
        fig.savefig(
            "%s.%s" % (outfile, outfig_format),
            format=outfig_format,
            bbox_inches="tight",
        )


def plot_gsi(
    data,
    x="lambda",
    y="GSI",
    hue="train_gender",
    fontsize=14,
    outfile=None,
    outfig_format=None,
):
    fig = plt.figure()
    plt.rcParams.update({"font.size": fontsize})
    sns.boxplot(data=data, x=x, y=y, hue=hue, showmeans=True)
    plt.legend(title="Target group")
    plt.rcParams["text.usetex"] = True
    plt.xlabel(r"$\lambda$")
    plt.rcParams["text.usetex"] = False
    plt.ylabel("Group Specificity Index (GSI)")
    if outfile is not None:
        savefig(fig, outfile, outfig_format)
    plt.show()


def plot_accuracy(
    data,
    x="Lambda",
    y="Accuracy",
    col="Target group",
    hue="Test set",
    style="Test set",
    kind="line",
    height=4,
    fontsize=14,
    outfile=None,
    outfig_format=None,
):
    fig = plt.figure()
    plt.rcParams.update({"font.size": fontsize})
    g = sns.relplot(
        data=data,
        x=x,
        y=y,
        col=col,
        hue=hue,
        style=style,
        kind=kind,
        errorbar=("sd", 1),
        height=height,
    )
    (
        g.map(plt.axhline, y=0.9, color=".7", dashes=(2, 1), zorder=0)
        .set_axis_labels(r"$\lambda$", "Test Accuracy")
        .set_titles("Target group: {col_name}")
        .tight_layout(w_pad=0)
    )
    if outfile is not None:
        savefig(fig, outfile, outfig_format)

    plt.show()


def load_result(dataset, root_dir, lambdas, seed_start, test_size=0.0):
    """load brain left/right classification results for a dataset

    Args:
        dataset (string): _description_
        root_dir (string): _description_
        lambdas (list): _description_
        seed_start (_type_): _description_
        test_size (float, optional): _description_. Defaults to 0.0.
    """
    res_dict = dict()
    res_list = []
    test_size_str = str(int(test_size * 10))
    for lambda_ in lambdas:
        res_dict[lambda_] = []

    for lambda_ in lambdas:
        if not isinstance(lambda_, str):
            lambda_str = str(int(lambda_))
        else:
            lambda_str = lambda_
        model_dir = os.path.join(root_dir, "lambda%s" % lambda_str)
        for seed_iter in range(51):
            random_state = seed_start - seed_iter
            res_fname = "results_%s_L%s_test_size0%s_Fisherz_%s.csv" % (
                dataset,
                lambda_str,
                test_size_str,
                random_state,
            )
            res_fpath = os.path.join(model_dir, res_fname)
            if os.path.exists(res_fpath):
                res_df = pd.read_csv(os.path.join(model_dir, res_fname))
                res_df["seed"] = random_state
                res_dict[lambda_].append(res_df)
                res_list.append(res_df)

    for lambda_ in lambdas:
        res_dict[lambda_] = pd.concat(res_dict[lambda_])

    res_df_all = pd.concat(res_list)
    res_df_all = res_df_all.reset_index(drop=True)

    return res_df_all


def reformat_results(res_df, test_sets, male_label=0):
    """reformat results dataframe to one accuracy per row

    Args:
        res_df (_type_): _description_
        test_sets (_type_): _description_
        male_label (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    res_reformat = {
        "Accuracy": [],
        "Test set": [],
        "Lambda": [],
        "Target group": [],
        "seed": [],
        "split": [],
        "fold": [],
        "Train session": [],
    }
    for idx_ in res_df.index:
        # print(idx_, res_df.iloc[idx_, 12])
        subset_ = res_df.loc[idx_, :]
        for test_set in test_sets:
            res_reformat["Accuracy"].append(subset_[test_set])
            res_reformat["Lambda"].append(subset_["lambda"])
            if "train_session" in subset_:
                res_reformat["Train session"].append(subset_["train_session"])
            else:
                res_reformat["Train session"].append(None)
            if "target_group" not in subset_:  # for GSP dataset
                _group = subset_["train_gender"]
            else:
                _group = subset_["target_group"]
            test_set_list = test_set.split("_")
            if _group == "Male" or _group == male_label:
                res_reformat["Target group"].append("Male")
                if "oc" in test_set_list or "tgt" in test_set_list:
                    res_reformat["Test set"].append("Female")
                else:
                    res_reformat["Test set"].append("Male")
            else:
                res_reformat["Target group"].append("Female")
                if "oc" in test_set_list or "tgt" in test_set_list:
                    res_reformat["Test set"].append("Male")
                else:
                    res_reformat["Test set"].append("Female")

            for key in ["seed", "split", "fold"]:
                res_reformat[key].append(subset_[key])
    return pd.DataFrame(res_reformat)


def label_gender(df):
    df = df.copy()
    gender_map = {"0": "Male", "1": "Female"}
    df["train_gender"] = df["train_gender"].astype(str).map(gender_map)
    return df


def plot_hcp(res_df_all, test_size, out_dir="figures"):
    res_df_all["GSI_is"] = 2 * (
        res_df_all["acc_ic_is"] * (res_df_all["acc_ic_is"] - res_df_all["acc_oc_is"])
    )
    res_df_all["GSI_os"] = 2 * (
        res_df_all["acc_ic_os"] * (res_df_all["acc_ic_os"] - res_df_all["acc_oc_os"])
    )

    res_df_is = reformat_results(res_df_all, ["acc_ic_is", "acc_oc_is"])
    res_df_os = reformat_results(res_df_all, ["acc_ic_os", "acc_oc_os"])

    res_df_all = label_gender(res_df_all)

    plot_accuracy(res_df_is, outfile=f"{out_dir}/Fig_2A", outfig_format=PLOT_FORMATS)
    plot_accuracy(res_df_os, outfile=f"{out_dir}/Fig_S1A", outfig_format=PLOT_FORMATS)

    plot_gsi(
        res_df_all,
        x="lambda",
        y="GSI_is",
        hue="train_gender",
        outfile=f"{out_dir}/Fig_2B",
        outfig_format=PLOT_FORMATS,
    )
    plot_gsi(
        res_df_all,
        x="lambda",
        y="GSI_os",
        hue="train_gender",
        outfile=f"{out_dir}/Fig_S1B",
        outfig_format=PLOT_FORMATS,
    )

    if test_size == 0.2:
        res_df_test_sub = reformat_results(
            res_df_all, ["acc_tgt_test_sub", "acc_nt_test_sub"]
        )
        plot_accuracy(
            res_df_test_sub, outfile=f"{out_dir}/Fig_S1C", outfig_format=PLOT_FORMATS
        )
        res_df_all["GSI_test_sub"] = 2 * (
            res_df_all["acc_tgt_test_sub"]
            * (res_df_all["acc_tgt_test_sub"] - res_df_all["acc_nt_test_sub"])
        )
        plot_gsi(
            res_df_all,
            x="lambda",
            y="GSI_test_sub",
            hue="train_gender",
            outfile=f"{out_dir}/Fig_S1D",
            outfig_format=PLOT_FORMATS,
        )


def plot_gsp(res_df_all, test_size, out_dir="figures"):
    res_df_all["gap"] = res_df_all["acc_ic"] - res_df_all["acc_oc"]
    res_df_all["GSI"] = 2 * (
        res_df_all["acc_ic"] * (res_df_all["acc_ic"] - res_df_all["acc_oc"])
    )

    res_df_reformat = reformat_results(res_df_all, ["acc_ic", "acc_oc"])

    res_df_all = label_gender(res_df_all)

    suffix = {0.0: "S2A", 0.2: "S2C"}.get(test_size, "S2X")  # fallback suffix
    suffix_gsi = {0.0: "S2B", 0.2: "S2D"}.get(test_size, "S2Y")

    plot_accuracy(
        res_df_reformat, outfile=f"{out_dir}/Fig_{suffix}", outfig_format=PLOT_FORMATS
    )
    plot_gsi(
        res_df_all,
        x="lambda",
        y="GSI",
        hue="train_gender",
        outfile=f"{out_dir}/Fig_{suffix_gsi}",
        outfig_format=PLOT_FORMATS,
    )


def main():
    model_root_dir = "cls_results"
    seed_start = 2023
    datasets = {"gsp": [0.0, 0.2], "HCP": [0.0, 0.2]}
    out_dir = "cls_figures"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]

    for dataset in datasets.keys():
        result_dir = f"{model_root_dir}/{dataset}/"
        for test_size in datasets[dataset]:
            res_df_all = load_result(
                dataset=dataset,
                root_dir=result_dir,
                lambdas=lambdas,
                seed_start=seed_start,
                test_size=test_size,
            )

            if dataset == "HCP":
                plot_hcp(res_df_all, test_size, out_dir=out_dir)
            elif dataset == "gsp":
                plot_gsp(res_df_all, test_size, out_dir=out_dir)


if __name__ == "__main__":
    main()
