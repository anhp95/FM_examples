# %%
import os
import sys
import numpy as np
import torch
import tqdm

import terratorch
from terratorch.datamodules import MultiTemporalCropClassificationDataModule
from terratorch.tasks import SemanticSegmentationTask
from terratorch.datasets.transforms import (
    FlattenTemporalIntoChannels,
    UnflattenTemporalFromChannels,
)
from terratorch.registry import BACKBONE_REGISTRY

import albumentations
from albumentations import Compose, Flip
from albumentations.pytorch import ToTensorV2

import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint


# %%
DATASET_PATH = "C://prithvi_data"
# %%
from huggingface_hub import snapshot_download

repo_id = "ibm-nasa-geospatial/multi-temporal-crop-classification"
_ = snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    cache_dir="./cache",
    resume_download=True,
    local_dir=DATASET_PATH,
)
# %%
OUT_DIR = "./tiny_multicrop"  # where to save checkpoints and log files

BATCH_SIZE = 2
EPOCHS = 50
LR = 2.0e-4
WEIGHT_DECAY = 0.1
HEAD_DROPOUT = 0.1
FREEZE_BACKBONE = False

BANDS = ["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"]
NUM_FRAMES = 3

CLASS_WEIGHTS = [
    0.386375,
    0.661126,
    0.548184,
    0.640482,
    0.876862,
    0.925186,
    3.249462,
    1.542289,
    2.175141,
    2.272419,
    3.062762,
    3.626097,
    1.198702,
]

SEED = 0

# %%
# Adding augmentations for a temporal dataset requires additional transforms
train_transforms = [
    terratorch.datasets.transforms.FlattenTemporalIntoChannels(),
    albumentations.Flip(),
    albumentations.pytorch.transforms.ToTensorV2(),
    terratorch.datasets.transforms.UnflattenTemporalFromChannels(
        n_timesteps=NUM_FRAMES
    ),
]
# %%
# This datamodule allows access to the dataset in its various splits.
data_module = MultiTemporalCropClassificationDataModule(
    data_root=DATASET_PATH,
    train_transform=train_transforms,
    expand_temporal_dimension=True,
)
# %%
# Checking the dataset means and stds
data_module.means, data_module.stds
# %%
# Checking train split size
data_module.setup("fit")
train_dataset = data_module.train_dataset
len(train_dataset)
# %%
# Checking available bands
train_dataset.all_band_names
# %%
# Checking dataset classes
train_dataset.class_names
# %%
# Ploting a few samples
for i in range(5):
    train_dataset.plot(train_dataset[i])

# %%
# Checking validation split size
val_dataset = data_module.val_dataset
len(val_dataset)
# %%
# Checking test split
data_module.setup("test")
test_dataset = data_module.test_dataset
len(test_dataset)
# %%
pl.seed_everything(SEED)

# Logger
logger = TensorBoardLogger(
    save_dir=OUT_DIR,
    name="multicrop_example",
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val/mIoU",
    mode="max",
    dirpath=os.path.join(OUT_DIR, "multicrop_example", "checkpoints"),
    filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
)

# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    precision="bf16-mixed",
    num_nodes=1,
    logger=logger,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    limit_predict_batches=1,  # predict only in the first batch for generating plots
    num_sanity_val_steps=0,
)

# DataModule
data_module = MultiTemporalCropClassificationDataModule(
    batch_size=BATCH_SIZE,
    data_root=DATASET_PATH,
    train_transform=train_transforms,
    reduce_zero_label=True,
    expand_temporal_dimension=True,
    num_workers=0,
    use_metadata=True,
)
# %%
# Model

backbone_args = dict(
    backbone_pretrained=True,
    backbone="prithvi_eo_v2_600_tl",  # prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
    backbone_coords_encoding=["time", "location"],
    backbone_bands=BANDS,
    backbone_num_frames=NUM_FRAMES,
)

decoder_args = dict(
    decoder="UperNetDecoder",
    decoder_channels=256,
    decoder_scale_modules=True,
)

necks = [
    dict(
        name="SelectIndices",
        # indices=[2, 5, 8, 11],  # indices for prithvi_vit_100
        # indices=[5, 11, 17, 23],  # indices for prithvi_eo_v2_300
        indices=[7, 15, 23, 31],  # indices for prithvi_eo_v2_600
    ),
    dict(
        name="ReshapeTokensToImage",
        effective_time_dim=NUM_FRAMES,
    ),
]

model_args = dict(
    **backbone_args,
    **decoder_args,
    num_classes=len(CLASS_WEIGHTS),
    head_dropout=HEAD_DROPOUT,
    necks=necks,
    rescale=True,
)


model = SemanticSegmentationTask(
    model_args=model_args,
    plot_on_val=False,
    class_weights=CLASS_WEIGHTS,
    loss="ce",
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    ignore_index=-1,
    freeze_backbone=FREEZE_BACKBONE,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
)

# %%
trainer.fit(model, datamodule=data_module)
# %%
ckpt_path = checkpoint_callback.best_model_path

# Test results
test_results = trainer.test(model, datamodule=data_module, ckpt_path=ckpt_path)
# %%
preds = trainer.predict(model, datamodule=data_module, ckpt_path=ckpt_path)

# %%
data_loader = trainer.predict_dataloaders
batch = next(iter(data_loader))
# %%
for i in range(BATCH_SIZE):

    sample = {key: batch[key][i] for key in batch}
    sample["prediction"] = preds[0][0][0][i].cpu().numpy()

    data_module.predict_dataset.plot(sample)
# %%
