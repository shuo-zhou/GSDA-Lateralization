from ._artl import ARRLS, ARSVM
from ._code import CoDeLR
from ._gsda import CoDeLR_Torch, GSLR, hsic
from ._manifold_learn import LapRLS, LapSVM
from ._sider import SIDeRLS, SIDeRSVM

__all__ = [ARSVM, ARRLS, GSLR, hsic, LapSVM, LapRLS, SIDeRSVM, SIDeRLS, CoDeLR]
