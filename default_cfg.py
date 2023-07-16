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
_C.DATASET.ROOT = "/shared/tale2/Shared/data/HCP/BNA/Proc"
_C.DATASET.ATLAS = "BNA"
_C.DATASET.FEATURE = "correlation"
_C.DATASET.TEST_SIZE = 300
_C.DATASET.TYPE = "structural"
_C.DATASET.SESSIONS = [None]
_C.DATASET.RUN = "Fisherz"
_C.DATASET.CONNECTION = "intra"
_C.DATASET.NUM_REPEAT = 5
_C.DATASET.MIX_GEND = False
# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.SEED = 2021
_C.SOLVER.L2PARAM = 10  # Initial learning rate
_C.SOLVER.LAMBDA_ = 2.0  # Initial learning rate

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = CN()
_C.OUTPUT.ROOT = "/shared/tale2/Shared/data/HCP/BNA/Results"


def get_cfg_defaults():
    return _C.clone()
