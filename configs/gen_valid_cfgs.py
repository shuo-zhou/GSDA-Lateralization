import os

import numpy as np
from cr8_cfgs import mk_dir


def main():
    # dataset = "ukb"
    dataset = "gsp"
    hpc_node = "tale"
    # hpc_node = "rse"
    memory = 2
    batch_dir = os.path.join("/shared/tale2/Shared/szhou/qsub/Brain_LR", dataset)
    cfg_dir = os.path.join(batch_dir, "configs")
    data_dir = "/shared/tale2/Shared/data/brain/%s/proc" % dataset
    output_dir = "/shared/tale2/Shared/data/brain/%s/Models" % dataset
    for _dir in [batch_dir, cfg_dir]:
        mk_dir(_dir)
    py_dir = "/shared/tale2/Shared/szhou/code/Brain_LR_BNU"
    py_file = "main_validset.py"

    seeds = 2023 - np.arange(50)
    lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]



    for lambda_ in lambdas:
        qbatch_fname = "q_%s.sh" % lambda_
        qbatch_file = open(os.path.join(batch_dir, qbatch_fname), "w")
        for seed in seeds:
            cfg_fname = "%s_lambda%s_%s.yaml" % (dataset, lambda_, seed)
            cfg_file = open(os.path.join(cfg_dir, cfg_fname), "w")
            cfg_file.write("DATASET:\n")
            cfg_file.write("  DATASET: %s\n" % dataset)
            cfg_file.write("  ROOT: %s\n" % data_dir)
            cfg_file.write("SOLVER:\n")
            cfg_file.write("  SEED: %s\n" % seed)
            cfg_file.write("  LAMBDA_: %s\n" % lambda_)
            cfg_file.write("OUTPUT:\n")
            cfg_file.write("  ROOT: %s\n" % output_dir)
            cfg_file.close()

            batch_fname = "%s_lambda%s_%s.sh" % (dataset, lambda_, seed)
            batch_file = open(os.path.join(batch_dir, batch_fname), "w")
            batch_file.write("#!/bin/bash\n")
            batch_file.write("#$ -P %s\n" % hpc_node)
            batch_file.write("#$ -q %s.q\n" % hpc_node)
            batch_file.write("#$ -l rmem=%sG\n\n" % memory)

            batch_file.write("module load apps/python/conda\n")
            batch_file.write("source activate pykale\n")
            batch_file.write("cd %s\n" % py_dir)
            batch_file.write("python %s --cfg %s/%s" % (py_file, cfg_dir, cfg_fname))
            batch_file.close()

            qbatch_file.write("qsub %s/%s\n" % (batch_dir, batch_fname))

        qbatch_file.close()


if __name__ == '__main__':
    main()