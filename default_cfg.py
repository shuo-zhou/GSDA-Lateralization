"""
Default configurations
"""

from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATASET = "HCP"
_C.DATASET.DOWNLOAD = True
_C.DATASET.ROOT = "data"
_C.DATASET.ATLAS = "BNA"
_C.DATASET.FEATURE = "correlation"
_C.DATASET.TEST_SIZE = 0.2
_C.DATASET.TYPE = "functional"
_C.DATASET.SESSIONS = [None]
_C.DATASET.RUN = "Fisherz"
_C.DATASET.CONNECTION = "intra"
_C.DATASET.NUM_REPEAT = 5
_C.DATASET.MIX_GROUP = False
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2023
_C.SOLVER.L2PARAM = 10  # Initial learning rate
_C.SOLVER.LAMBDA_ = [0.0, 1.0, 2.0, 5.0, 8.0, 10.0]

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "/shared/tale2/Shared/data/HCP/BNA/Results"


def get_cfg_defaults():
    return _C.clone()
