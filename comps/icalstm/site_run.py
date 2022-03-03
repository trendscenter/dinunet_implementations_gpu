import sys
sys.path.append('.')
from coinstac_dinunet.site_runner import SiteRunner

from comps.icalstm import ICATrainer, ICADataHandle, ICADataset

if __name__ == "__main__":
    runner = SiteRunner(taks_id='ICA', data_path='../../datasets/icalstm', mode='Train', split_ratio=[0.8, 0, 0.2])
    runner.run(ICATrainer, ICADataset, ICADataHandle)
