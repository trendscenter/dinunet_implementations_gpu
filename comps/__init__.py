from enum import Enum

from .vbm import *


class NNComputation(str, Enum):
    """ Available tasks """
    TASK_VBM = "VBM-Classification"


class AggEngine(str, Enum):
    DECENTRALIZED_SGD = "dSGD"
