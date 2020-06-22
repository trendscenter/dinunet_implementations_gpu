import os
import numpy as np
import pandas as pd
import traceback
import sys
import os


def parse_subj_volume_files(dir, files=None):
    data, errors = None, []
    for file in os.listdir(dir) if not files else files:
        try:
            df = pd.read_csv(dir + os.sep + file, sep='\t', names=['File', file], skiprows=1)
            df = df.set_index(df.columns[0])
            data = df if data is None else data.join(df)
        except:
            errors.append([file, sys.exc_info()])
    return data.T, errors
