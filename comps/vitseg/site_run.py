from coinstac_dinunet.site_runner import SiteRunner

from comps.vitseg import VitSegTrainer, BrainSegDataHandle, BrainSegDataset

if __name__ == "__main__":
    runner = SiteRunner(taks_id='VITSEG', data_path='../../datasets/vitseg', mode='Train', seed=10, site_index=1,
                        split_ratio=[0.6, 0.2, 0.2], monitor_metric='dice', log_header='Loss|DICE', batch_size=16,
                        )
    runner.run(VitSegTrainer, BrainSegDataset, BrainSegDataHandle)
