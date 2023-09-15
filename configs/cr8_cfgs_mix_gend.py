import os

import numpy as np
from cr8_cfgs import mk_dir


def main():
    # dataset = "ukb"
    dataset = "gsp"
    data_dir = "/shared/tale2/Shared/data/brain/%s/proc" % dataset
    py_dir = "/shared/tale2/Shared/szhou/code/Brain_LR_BNU"
    batch_dir = "/shared/tale2/Shared/szhou/qsub/Brain_LR/%s" % dataset
    output_dir = "/shared/tale2/Shared/data/brain/%s/Models" % dataset
    cfg_dir = os.path.join(batch_dir, "configs")
    for _dir in [batch_dir, cfg_dir]:
        mk_dir(_dir)
    # py_file = "main.py"
    py_file = "main_validset.py"
    seeds = 2023 - np.arange(50)

    lambda_ = 0.0
    qbatch_fname = "q_mix_gend_%s.sh" % dataset
    qbatch_file = open(os.path.join(batch_dir, qbatch_fname), "w")
    for seed in seeds:
        cfg_fname = "lambda%s_%s_mix_gend.yaml" % (lambda_, seed)
        cfg_file = open(os.path.join(cfg_dir, cfg_fname), "w")
        cfg_file.write("DATASET:\n")
        cfg_file.write("  DATASET: %s\n" % dataset)
        cfg_file.write("  ROOT: %s\n" % data_dir)
        cfg_file.write("  MIX_GEND: True\n")
        cfg_file.write("SOLVER:\n")
        cfg_file.write("  SEED: %s\n" % seed)
        cfg_file.write("  LAMBDA_: %s\n" % lambda_)
        cfg_file.write("OUTPUT:\n")
        cfg_file.write("  ROOT: %s\n" % output_dir)
        cfg_file.close()

        batch_fname = "lambda%s_%s_mix_gend.sh" % (lambda_, seed)
        batch_file = open(os.path.join(batch_dir, batch_fname), "w")
        batch_file.write("#!/bin/bash\n")
        batch_file.write("#$ -P tale\n")
        batch_file.write("#$ -q tale.q\n")
        batch_file.write("#$ -l rmem=3G\n\n")

        batch_file.write("module load apps/python/conda\n")
        batch_file.write("source activate pykale\n")
        batch_file.write("cd %s\n" % py_dir)
        batch_file.write("python %s --cfg %s/%s" % (py_file, cfg_dir, cfg_fname))
        batch_file.close()

        qbatch_file.write("qsub %s/%s\n" % (batch_dir, batch_fname))

    qbatch_file.close()


if __name__ == '__main__':
    main()
