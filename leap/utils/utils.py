from pathlib import Path

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


def normalize(df, feat_cols, label_cols, method, path, reverse=False, eps=1e-8):
    if method == "standard":
        if feat_cols is not None:
            x_mat = df.select(feat_cols).to_numpy()
            x_mean = pl.read_parquet(Path(path, "x_mean.parquet")).select(feat_cols).to_numpy()
            x_std = pl.read_parquet(Path(path, "x_std.parquet")).select(feat_cols).to_numpy()
            if reverse:
                x_mat = x_mat * x_std + x_mean
            else:
                x_std = x_std.clip(eps)
                x_mat = (x_mat - x_mean) / x_std
        if label_cols is not None:
            y_mat = df.select(label_cols).to_numpy()
            y_mean = pl.read_parquet(Path(path, "y_mean.parquet")).select(label_cols).to_numpy()
            y_std = pl.read_parquet(Path(path, "y_std.parquet")).select(label_cols).to_numpy()
            if reverse:
                y_mat = y_mat * y_std + y_mean
            else:
                y_std = y_std.clip(eps)
                y_mat = (y_mat - y_mean) / y_std
    elif method == "robust":
        if feat_cols is not None:
            x_mat = df.select(feat_cols).to_numpy()
            x_q2 = pl.read_parquet(Path(path, "x_q2.parquet")).select(feat_cols).to_numpy()
            x_q3 = pl.read_parquet(Path(path, "x_q3.parquet")).select(feat_cols).to_numpy()
            x_q4 = pl.read_parquet(Path(path, "x_q4.parquet")).select(feat_cols).to_numpy()
            if reverse:
                x_mat = x_mat * (x_q4 - x_q2) + x_q3
            else:
                x_mat = (x_mat - x_q3) / np.maximum(x_q4 - x_q2, eps)
        if label_cols is not None:
            y_mat = df.select(label_cols).to_numpy()
            y_q2 = pl.read_parquet(Path(path, "y_q2.parquet")).select(label_cols).to_numpy()
            y_q3 = pl.read_parquet(Path(path, "y_q3.parquet")).select(label_cols).to_numpy()
            y_q4 = pl.read_parquet(Path(path, "y_q4.parquet")).select(label_cols).to_numpy()
            if reverse:
                y_mat = y_mat * (y_q4 - y_q2) + y_q3
            else:
                y_mat = (y_mat - y_q3) / np.maximum(y_q4 - y_q2, eps)
    else:
        raise NotImplementedError
    if feat_cols is not None:
        df = df.with_columns(
            [
                pl.lit(x_mat[:, i]).alias(col)
                for i, col in enumerate(feat_cols)
            ]
        )
    if label_cols is not None:
        df = df.with_columns(
            [
                pl.lit(y_mat[:, i]).alias(col)
                for i, col in enumerate(label_cols)
            ]
        )
    return df
