import argparse

from default_cfg import get_cfg_defaults
from experiment import run_experiment


def arg_parse():
    """Parsing arguments"""
    parser = argparse.ArgumentParser(
        description="GSDA brain hemispheres classification"
    )
    parser.add_argument("--cfg", required=True, help="path to config file", type=str)
    # parser.add_argument("--gpus", default=None, help="gpu id(s) to use", type=str)
    parser.add_argument("--resume", default="", type=str)
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    # ---- setup configs ----
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    results, outfile = run_experiment(cfg)
    results.to_csv(outfile, index=False)


if __name__ == "__main__":
    main()
