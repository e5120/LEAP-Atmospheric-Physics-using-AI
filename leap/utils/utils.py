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


def normalize(df, feat_cols, label_cols, method, path, reverse=False, eps=1e-10):
    if feat_cols:
        x_stats = pl.read_parquet(Path(path, "feat_stats.parquet")).select(feat_cols).to_numpy()
        x_mat = df.select(feat_cols).to_numpy()
    if label_cols:
        y_stats = pl.read_parquet(Path(path, "label_stats.parquet")).select(label_cols).to_numpy()
        y_mat = df.select(label_cols).to_numpy()
    if method == "standard":
        if feat_cols:
            x_mean, x_std = x_stats[0], x_stats[1]
            x_mat = standard_scale(x_mat, x_mean, x_std, reverse=reverse, eps=eps)
        if label_cols:
            y_mean, y_std = y_stats[0], y_stats[1]
            y_mat = standard_scale(y_mat, y_mean, y_std, reverse=reverse, eps=eps)
    elif method == "robust":
        if feat_cols:
            x_q2, x_q3, x_q4 = x_stats[2], x_stats[3], x_stats[4]
            x_mat = robust_scale(x_mat, x_q2, x_q3, x_q4, reverse=reverse, eps=eps)
        if label_cols:
            y_q2, y_q3, y_q4 = y_stats[2], y_stats[3], y_stats[4]
            y_mat = robust_scale(y_mat, y_q2, y_q3, y_q4)
    else:
        raise NotImplementedError
    if feat_cols:
        df = df.with_columns(
            [
                pl.lit(x_mat[:, i]).alias(col)
                for i, col in enumerate(feat_cols)
            ]
        )
    if label_cols:
        df = df.with_columns(
            [
                pl.lit(y_mat[:, i]).alias(col)
                for i, col in enumerate(label_cols)
            ]
        )
    return df


def standard_scale(mat, mean, std, reverse=False, eps=1e-10):
    if reverse:
        mat = mat * std + mean
    else:
        std = std.clip(eps)
        mat = (mat - mean) / std
    return mat


def robust_scale(mat, q2, q3, q4, reverse=False, eps=1e-10):
    if reverse:
        mat = mat * (q4 - q2) + q3
    else:
        mat = (mat - q3) / np.maximum(q4 - q2, eps)
    return mat
