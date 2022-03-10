from enum import Enum

from .fs import *
from .icalstm import *
from .vbm import *


class NNComputation(str, Enum):
    """ Available tasks """
    TASK_VBM = "VBM-Classification"
    TASK_FREESURFER = "FS-Classification"
    TASK_ICA = "ICA-Classification"


class AggEngine(str, Enum):
    DECENTRALIZED_SGD = "dSGD"
    RANK_DAD = "rankDAD"
    POWER_SGD = "powerSGD"
