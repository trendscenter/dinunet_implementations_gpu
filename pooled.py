from coinstac_dinunet import PooledTrainer

from classification import NiftiTrainer, NiftiDataset

if __name__ == "__main__":
    trainer = PooledTrainer(NiftiTrainer,
                            mode='train', model_scale=4,
                            epochs=21, batch_size=16, patience=31,
                            learning_rate=0.001, gpus=[0])
    trainer.run(NiftiDataset, only_sites=None, only_folds=None)
