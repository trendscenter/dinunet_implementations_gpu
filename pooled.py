from coinstac_dinunet import PooledTrainer

from nn_implementations.vbm import VBMTrainer, VBMDataset

if __name__ == "__main__":
    trainer = PooledTrainer(VBMTrainer,
                            mode='train', model_scale=4,
                            epochs=21, batch_size=16, patience=31,
                            learning_rate=0.001, gpus=[0])
    trainer.run(VBMDataset, only_sites=None, only_folds=None)
