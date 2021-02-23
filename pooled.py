from coinstac_dinunet import PooledTrainer

from local import NiftiTrainer, NiftiDataset
import coinstac_dinunet.config as cfg

if __name__ == "__main__":
    trainer = PooledTrainer(NiftiTrainer,
                            mode='test',
                            epochs=251, batch_size=4, patience=31,
                            learning_rate=0.0002, gpus=[])
    trainer.run(NiftiDataset, only_sites=None, only_folds=None)
