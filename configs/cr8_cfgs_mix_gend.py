import os
import numpy as np


def mk_dir(dir_):
    if not os.path.exists(dir_):
        os.mkdir(dir_)


py_dir = "/shared/tale2/Shared/szhou/code/Brain_LR_BNU"
batch_dir = "/shared/tale2/Shared/szhou/qsub/Brain_LR"
py_file = "main_hpc.py"
seeds = 2022 - np.arange(50)
lambdas = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]

cfg_dir = os.path.join(batch_dir, "configs")

lambda_ = 0.0
qbatch_fname = "q_mix_gend.sh"
qbatch_file = open(os.path.join(batch_dir, qbatch_fname), "w")
for seed in seeds:
    cfg_fname = "lambda%s_%s_mix_gend.yaml" % (lambda_, seed)
    cfg_file = open(os.path.join(cfg_dir, cfg_fname), "w")
    cfg_file.write("DATASET:\n")
    cfg_file.write("  MIX_GEND: True\n")
    cfg_file.write("SOLVER:\n")
    cfg_file.write("  SEED: %s\n" % seed)
    cfg_file.write("  LAMBDA_: %s\n" % lambda_)
    cfg_file.close()

    batch_fname = "lambda%s_%s_mix_gend.sh" % (lambda_, seed)
    batch_file = open(os.path.join(batch_dir, batch_fname), "w")
    batch_file.write("#!/bin/bash\n")
    batch_file.write("#$ -P rse\n")
    batch_file.write("#$ -q rse.q\n")
    batch_file.write("#$ -l rmem=8G\n\n")

    batch_file.write("module load apps/python/anaconda3-4.2.0\n")
    batch_file.write("source activate pykale\n")
    batch_file.write("cd %s\n" % py_dir)
    batch_file.write("python %s --cfg %s/%s" % (py_file, cfg_dir, cfg_fname))
    batch_file.close()

    qbatch_file.write("qsub %s/%s\n" % (batch_dir, batch_fname))

qbatch_file.close()