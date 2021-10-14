from enum import Enum

from .vbm import *
from .fs import *


class NNComputation(str, Enum):
    """ Available tasks """
    TASK_VBM = "VBM-Classification"
    TASK_FREESURFER = "FS-Classification"


class AggEngine(str, Enum):
    DECENTRALIZED_SGD = "dSGD"
    RANK_DAD = "rankDAD"
