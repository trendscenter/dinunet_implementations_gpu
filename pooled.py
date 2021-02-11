from coinstac_dinunet import PooledTrainer

from local import NiftiTrainer, NiftiDataset

if __name__ == "__main__":
    trainer = PooledTrainer(NiftiTrainer,
                            mode='train',
                            # pretrained_path='net_logs/weights.tar',
                            epochs=51, batch_size=32, patience=11,
                            learning_rate=0.001)
    trainer.cache['gpus'] = []
    trainer.run(NiftiDataset, only_sites=None, only_folds=[0])
