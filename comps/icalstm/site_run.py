from coinstac_dinunet.site_runner import SiteRunner

from . import ICATrainer, ICADataHandle, ICADataset

if __name__ == "__main__":
    runner = SiteRunner(taks_id='FSL', data_path='../../datasets/ica_s2', mode='Train', split_ratio=[0.8, 0.1, 0.1])
    runner.run(ICATrainer, ICADataset, ICADataHandle)
