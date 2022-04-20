from typing import List
from pathlib import Path

import argparse
import collections
from collections import OrderedDict

from brain_dataset_cache import BrainCacheDataset
from model import MeshNet, UNet
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from catalyst import metrics
from catalyst.callbacks import CheckpointCallback
from catalyst.contrib.utils.pandas import dataframe_to_list
from catalyst.data import BatchPrefetchLoaderWrapper, ReaderCompose
from catalyst.dl import Runner, DeviceEngine, DataParallelEngine, DataParallelAMPEngine, AMPEngine
from catalyst.metrics.functional._segmentation import dice
from torch.utils.checkpoint import checkpoint_sequential

from monai.data import ThreadDataLoader, DataLoader
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    SpatialPadd,
    RandAffined,
    Rand3DElasticd,
    RandAdjustContrastd,
    EnsureTyped,
    ToDeviced,
)

from PVT.segmentation.pvt2 import pvt3d_tiny, pvt3d_small, pvt3d_medium, pvt3d_large
from meshvit.ViT3d import build_vit
from segmenter.segm.model.decoder3d import MaskTransformer3d
from segmenter.segm.model.segmenter3d import Segmenter3d
from segmenter.segm.model.vit3d import VisionTransformer3d

def voxel_majority_predict_from_subvolumes(
    loader, n_classes
):
    """
    # TODO change function to allow for cuda or cpu based predictions with cuda
    # as default.
    Predicts Brain Segmentations given a dataloader class and a optional dict
    to contain the outputs. Returns a dict of brain segmentations.
    """
    subject_macro_dice = []
    subject_micro_dice = []
    prediction_n = 0

    segmentation = torch.zeros(
        tuple(np.insert(loader.dataset.generator.volume_shape, 0, n_classes)),
        dtype=torch.uint8).cpu()

    for inference in tqdm(runner.predict_loader(loader=loader)):
        coords = inference[1]
        _, predicted = torch.max(F.log_softmax(inference[0], dim=1), 1)

        for j in range(predicted.shape[0]):
            c_j = coords[j][0]
            subj_id = prediction_n // loader.dataset.n_subvolumes
            labels = nib.load(loader.dataset.data[subj_id]['nii_labels']).get_fdata()

            for c in range(n_classes):
                segmentation[
                    c,
                    c_j[0, 0] : c_j[0, 1],
                    c_j[1, 0] : c_j[1, 1],
                    c_j[2, 0] : c_j[2, 1],
                ] += (predicted[j] == c).to('cpu')
            prediction_n += 1

            if (prediction_n // loader.dataset.n_subvolumes) > subj_id:
                seg = torch.max(segmentation, 0)[1]
                seg = torch.nn.functional.one_hot(
                    seg, args.n_classes).permute(0, 3, 1, 2)
                one_hot_label = torch.nn.functional.one_hot(
                    torch.from_numpy(labels).long(),
                                     args.n_classes).permute(0, 3, 1, 2)
                subject_macro_dice.append(dice(
                    seg,
                    one_hot_label,
                    mode='macro').item())

                subject_micro_dice.append(dice(
                    seg,
                    one_hot_label).detach().numpy())

                segmentation = torch.zeros(
                    tuple(np.insert(loader.dataset.generator.volume_shape, 0, n_classes)),
                    dtype=torch.uint8).cpu()

    macro_dice_df = pd.DataFrame({'macro_dice': subject_macro_dice})
    micro_dice_df = pd.DataFrame(np.stack(subject_micro_dice))

    return macro_dice_df, micro_dice_df


def get_loaders(
    random_state: int,
    volume_shape: List[int],
    subvolume_shape: List[int],
    train_subvolumes: int = 128,
    infer_subvolumes: int = 32,
    in_csv_train: str = None,
    in_csv_valid: str = None,
    in_csv_infer: str = None,
    batch_size: int = 2,
    num_workers: int = 64,
) -> dict:
    """Get Dataloaders"""


    loaders = collections.OrderedDict()
    datasets = {}
    load_image_trans = LoadImaged(keys=["images", "nii_labels"])
    add_channel = AddChanneld(keys=["images", "nii_labels"])
    pad = SpatialPadd(keys=["images", "nii_labels"], spatial_size=[256, 256, 256])
    min_max_scale = ScaleIntensityd(keys=["images"])
    transform_chance = .5
    affine = RandAffined(keys=['images', 'nii_labels'], prob=transform_chance,
                         rotate_range=0.8,
                         translate_range=.3,
                         shear_range=0.1,
                         scale_range=0.2,
                         cache_grid=True,
                         device=torch.device("cuda"))
    #elastic very slow currently
    #elastic = Rand3DElasticd(keys=['images'], prob=transform_chance,
    #                         sigma_range=(1, 2),
    #                         magnitude_range=(1, 1),
    #                         device=torch.device("cuda"))

    to_device = ToDeviced(keys=['images', 'nii_labels'], device='cuda')


    typed = EnsureTyped(keys=["images", "nii_labels"])
    train_transforms = Compose([load_image_trans, add_channel, #min_max_scale, add_channel,
                                typed,
                                pad])#, to_device])

    #train_rand_transforms = Compose([to_device, affine])#, elastic])
    train_rand_transforms = Compose([to_device, affine])#, elastic])

    for mode, source in zip(
        ("train", "validation", "infer"),
        (in_csv_train, in_csv_valid, in_csv_infer),
    ):
        if mode == "infer":
            shuffle_flag = False
            n_subvolumes = infer_subvolumes

        else:
            n_subvolumes = train_subvolumes
            shuffle_flag = True

        if source is not None and len(source) > 0:
            data = dataframe_to_list(pd.read_csv(source))
            datasets[mode] = BrainCacheDataset(
                data,
                list_shape=volume_shape,
                list_sub_shape=subvolume_shape,
                n_subvolumes=n_subvolumes,
                mode=mode,
                num_workers=48,
                copy_cache=False,
                transform=train_transforms,
                rand_transform=train_rand_transforms,
            )

        loaders[mode] = ThreadDataLoader(dataset=datasets[mode],
                                         batch_size=batch_size,
                                         shuffle=shuffle_flag, num_workers=0)

    return loaders


class CustomRunner(Runner):
    """Custom Runner for demonstrating a NeuroImaging Pipeline"""

    def __init__(self, n_classes: int, parallel: bool, grad_checkpoint: bool):
        """Init."""
        super().__init__()
        self.n_classes = n_classes
        self.parallel = parallel
        self.grad_checkpoint = grad_checkpoint

    def get_engine(self):
        """Gets engine for multi or single gpu case"""
        if self.parallel:
            engine = DataParallelAMPEngine()

        else:
            engine = AMPEngine()

        return engine

    def get_loaders(self, stage: str) -> "OrderedDict[str, DataLoader]":
        """Returns the loaders for a given stage."""
        self._loaders = self._loaders
        return self._loaders

    def on_loader_start(self, runner):
        """
        Calls runner methods when the dataloader begins and adds
        metrics for loss and macro_dice
        """
        super().on_loader_start(runner)
        self.meters = {
            key: metrics.AdditiveValueMetric(compute_on_call=False)
            for key in ["loss", "macro_dice"]
        }

    def handle_batch(self, batch):
        """
        Custom train/ val step that includes batch unpacking, training, and
        DICE metrics
        """
        # model train/valid step
        x, y = batch["images"], batch["nii_labels"].long()

        with self.engine.autocast():
            if self.grad_checkpoint:
                segments = 4
                y_hat = checkpoint_sequential(self.model.module.model, segments, x)
            else:
                y_hat = self.model(x)

            ce_loss = F.cross_entropy(y_hat, y)

        one_hot_targets = (
            torch.nn.functional.one_hot(y, self.n_classes)
            .permute(0, 4, 1, 2, 3)
        )

        loss = ce_loss

        if self.is_train_loader:
            self.engine.backward_loss(loss, self.model, self.optimizer)
            self.engine.optimizer_step(loss, self.model, self.optimizer)
            #scheduler.step()
            self.optimizer.zero_grad()

        macro_dice = dice(F.softmax(y_hat), one_hot_targets, mode="macro")

        self.batch_metrics.update({"loss": loss, "macro_dice": macro_dice})

        for key in ["loss", "macro_dice"]:
            self.meters[key].update(
                self.batch_metrics[key].item(), self.batch_size
            )

    def on_loader_end(self, runner):
        """
        Calls runner methods when a dataloader finishes running and updates
        metrics
        """
        for key in ["loss", "macro_dice"]:
            self.loader_metrics[key] = self.meters[key].compute()[0]
        super().on_loader_end(runner)

    def predict_batch(self, batch):
        """
        Predicts a batch for an inference dataloader and returns the
        predictions as well as the corresponding slice indices
        """
        # model inference step
        input = batch["images"]
        input = self.engine.sync_device(input)

        if self.parallel:
            for layer in self.model.module.model:
                input = layer(input)

        else:
            for layer in self.model.model:
                input = layer(input)

        y_hat = input

        return (
            y_hat,
            batch["coords"],
        )


if __name__ == "__main__":
    import datetime
    parser = argparse.ArgumentParser(description="T1 segmentation Training")

    parser.add_argument(
        "--train_path",
        metavar="PATH",
        default="./data/dataset_train.csv",
        help="Path to list with brains for training",
    )
    parser.add_argument(
        "--validation_path",
        metavar="PATH",
        default="./data/dataset_valid.csv",
        help="Path to list with brains for validation",
    )
    parser.add_argument(
        "--inference_path",
        metavar="PATH",
        default="./data/dataset_infer.csv",
        help="Path to list with brains for inference",
    )
    parser.add_argument(
        "--logdir",
        metavar="PATH",
        default="/data/users2/bbaker43/MeshVit/meshnet_results/",
        help="Path to list with brains for validation",
    )
    parser.add_argument("--n_classes", default=31, type=int)
    parser.add_argument("--n_filters", default=None, type=int)
    parser.add_argument(
        "--train_subvolumes",
        default=128,
        type=int,
        metavar="N",
        help="Number of total subvolumes to sample from one brain",
    )
    parser.add_argument(
        '--patch_size',
        default=32,
        type=int,
        help="patch size for ViT embedding"
    )
    parser.add_argument(
        "--infer_subvolumes",
        default=512,
        type=int,
        metavar="N",
        help="Number of total subvolumes to sample from one brain",
    )
    parser.add_argument(
        "--sv_w", default=38, type=int, metavar="N", help="Width of subvolumes"
    )
    parser.add_argument(
        "--sv_h",
        default=38,
        type=int,
        metavar="N",
        help="Height of subvolumes",
    )
    parser.add_argument(
        "--sv_d", default=38, type=int, metavar="N", help="Depth of subvolumes"
    )
    parser.add_argument("--model", default="meshnet")
    parser.add_argument(
        "--dropout",
        default=0,
        type=float,
        metavar="N",
        help="dropout probability for meshnet",
    )
    parser.add_argument("--large", default=False)
    parser.add_argument("--parallel", default=False)
    parser.add_argument("--grad_checkpoint", default=False)
    parser.add_argument(
        "--n_epochs",
        default=30,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--vit_enc_n_layers",
        default=12
    )
    parser.add_argument(
        "--vit_enc_d_model",
        default=128
    )
    parser.add_argument(
        "--vit_enc_d_ff",
        default=128
    )
    parser.add_argument(
        "--vit_enc_n_heads",
        default=8
    )
    parser.add_argument(
        "--vit_dec_n_layers",
        default=2
    )
    parser.add_argument(
        "--vit_dec_n_heads",
        default=8
    )
    parser.add_argument(
        "--vit_dec_d_model",
        default=128
    )
    parser.add_argument(
        "--vit_dec_drop_path_rate",
        default=0.0
    )
    parser.add_argument(
        "--vit_dec_dropout",
        default=0.1
    )
    parser.add_argument(
        "--vit_dec_d_ff",
        default=128
    )
    args = parser.parse_args()
    print("{}".format(args))

    volume_shape = [256, 256, 256]
    subvolume_shape = [args.sv_h, args.sv_w, args.sv_d]

    loaders = get_loaders(
        0,
        volume_shape,
        subvolume_shape,
        args.train_subvolumes,
        args.infer_subvolumes,
        args.train_path,
        args.validation_path,
        args.inference_path
    )

    if args.model == "meshnet":
        net = MeshNet(
            n_channels=1,
            n_classes=args.n_classes,
            large=args.large,
            #n_filters=args.n_filters,
            dropout_p=args.dropout,
        )
    elif args.model == "segmenter":
        #net = build_vit(dim=128, patch_size=4, k=8, depth=12, heads=8, channels=1, output_shape=[args.sv_w, args.sv_w, args.sv_w], num_classes=args.n_classes)
        patch_size = args.patch_size
        vit_enc_n_layers = args.vit_enc_n_layers
        vit_enc_d_model = args.vit_enc_d_model
        vit_enc_d_ff = args.vit_enc_d_ff
        vit_enc_n_heads = args.vit_enc_n_heads
        vit = VisionTransformer3d((args.sv_w, args.sv_h, args.sv_h), patch_size, vit_enc_n_layers, vit_enc_d_model, vit_enc_d_ff, vit_enc_n_heads, args.n_classes, channels=1)
        vit_dec_n_layers = args.vit_dec_n_layers
        vit_dec_n_heads = args.vit_dec_n_heads
        vit_dec_d_model = args.vit_dec_d_model
        vit_dec_d_ff = args.vit_dec_d_ff
        drop_path_rate = args.drop_path_rate
        dropout = args.dropout
        decoder = MaskTransformer3d(args.n_classes, patch_size, vit_enc_d_ff, vit_dec_n_layers, vit_dec_n_heads, vit_dec_d_model, vit_dec_d_ff, drop_path_rate, dropout)
        net = Segmenter3d(vit, decoder, n_cls=args.n_classes)
    else:
        net = UNet(n_channels=1, n_classes=args.n_classes)

    logdir = "{log_dir}/{date}_{model}_{filters}_filters_gmwm_sv_{sv}_full_train".format(
        date=str(datetime.datetime.now().strftime("%d.%m.%Y.%H.%M.%S")), log_dir=args.logdir, model=args.model, filters=args.n_filters, sv=args.sv_w)

    if args.large:
        logdir += "_large"

    if args.dropout:
        logdir += "_dropout"

    optimizer = torch.optim.Adam(net.parameters(), lr=.001)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=.001,
        epochs=args.n_epochs,
        steps_per_epoch=len(loaders["train"]),
    )

    Path(logdir).mkdir(parents=True, exist_ok=True)

    runner = CustomRunner(n_classes=args.n_classes,
                          parallel=args.parallel,
                          grad_checkpoint = args.grad_checkpoint)

    runner.train(
        model=net,
        optimizer=optimizer,
        loaders={'train': loaders['train'],
                 'validation': loaders['validation']},
        num_epochs=args.n_epochs,
        scheduler=scheduler,
        callbacks=[CheckpointCallback(logdir=logdir)],
        logdir=logdir,
        verbose=True,
    )

    macro_dice_df, micro_dice_df = voxel_majority_predict_from_subvolumes(
        loaders['infer'], args.n_classes
    )

    macro_dice_df.to_csv('{logdir}/macro_dice_results.csv'.format(logdir=logdir), index=False)
    micro_dice_df.to_csv('{logdir}/micro_dice_results.csv'.format(logdir=logdir), index=False)

    print(macro_dice_df, micro_dice_df)
    print(macro_dice_df.mean())

