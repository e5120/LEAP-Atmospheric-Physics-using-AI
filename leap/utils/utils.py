import hydra
import polars as pl
import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)
from sklearn.model_selection import train_test_split


def setup(cfg):
    if cfg.model_checkpoint.dirpath:
        cfg.model_checkpoint.dirpath = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    else:
        cfg.model_checkpoint.dirpath = None
    L.seed_everything(cfg["seed"])
    if "benchmark" in cfg:
        torch.backends.cudnn.benchmark = cfg.benchmark


def get_num_training_steps(n_data, cfg):
    steps_per_epoch = n_data // cfg.batch_size // len(cfg.trainer.devices) // cfg.trainer.accumulate_grad_batches
    num_training_steps = steps_per_epoch * cfg.trainer.max_epochs
    return num_training_steps


def build_callbacks(cfg):
    checkpoint_callback = ModelCheckpoint(
        filename=f"model-{{val_r2:.4f}}",
        **cfg.model_checkpoint,
    )
    early_stop_callback = EarlyStopping(**cfg.early_stopping)
    progress_bar_callback = TQDMProgressBar(refresh_rate=1)
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        progress_bar_callback,
    ]
    return callbacks
