from coinstac_dinunet.site_runner import SiteRunner

from comps.icalstm import ICATrainer, ICADataHandle, ICADataset

if __name__ == "__main__":
    runner = SiteRunner(taks_id='ICA', data_path='../../datasets/icalstm', mode='train', seed=10, site_index=0,
                        split_ratio=[0.6, 0.2, 0.2], monitor_metric='auc', log_header='Loss|Auc', batch_size=32,
                        )
    runner.run(ICATrainer, ICADataset, ICADataHandle)
