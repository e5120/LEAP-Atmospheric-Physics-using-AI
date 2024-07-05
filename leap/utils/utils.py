from pathlib import Path

import hydra
import polars as pl
import numpy as np
import tensorflow as tf
import torch
import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TQDMProgressBar,
)


def setup(cfg):
    cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    if cfg.model_checkpoint.dirpath:
        cfg.model_checkpoint.dirpath = cfg.output_dir
    else:
        cfg.model_checkpoint.dirpath = None
    L.seed_everything(cfg["seed"])
    if "benchmark" in cfg:
        torch.backends.cudnn.benchmark = cfg.benchmark
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
    else:
        print("Not enough GPU hardware devices available")


def get_num_training_steps(n_data, cfg):
    num_devices = 1 if isinstance(cfg.trainer.devices, int) else len(cfg.trainer.devices)
    steps_per_epoch = n_data // cfg.batch_size // num_devices // cfg.trainer.accumulate_grad_batches
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


def normalize(df, methods, path, reverse=False, eps=1e-52):
    stats_df = pl.read_parquet(Path(path, "stats.parquet"))
    for method_name, columns in methods.items():
        if columns is None or len(columns) == 0:
            continue
        tgt_columns = []
        for col in columns:
            if col in df.columns:
                tgt_columns.append(col)
            else:
                tgt_columns += df.select(pl.col(f"^{col}_\d+$")).columns
        if len(tgt_columns) == 0:
            continue
        stats = stats_df.select(tgt_columns).to_numpy()
        mat = df.select(tgt_columns).to_numpy()
        mean, std, q2, q3, q4, mi, ma, lam, std_y = stats
        if method_name == "standard":
            mat = standard_scale(mat, mean, std, reverse=reverse, eps=eps)
        elif method_name == "robust":
            mat = robust_scale(mat, q2, q3, q4, reverse=reverse, eps=eps)
        elif method_name == "minmax":
            mat = minmax_scale(mat, mi, ma, reverse=reverse, eps=eps)
        elif method_name == "minmaxmean":
            mat = robust_scale(mat, mi, mean, ma, reverse=reverse, eps=eps)
        elif method_name == "minmaxzero":
            mat = robust_scale(mat, mi, 0, ma, reverse=reverse, eps=eps)
        elif method_name == "exp":
            mat = exp_scale(mat, lam, reverse=reverse, eps=eps)
        elif method_name == "std":
            mat = standard_scale(mat, 0, std, reverse=reverse, eps=eps)
        elif method_name == "standard_y":
            mat = standard_scale(mat, mean, std_y, reverse=reverse, eps=eps)
        else:
            raise NotImplementedError
        df = df.with_columns([
            pl.lit(mat[:, i]).alias(col)
            for i, col in enumerate(tgt_columns)
        ])
    return df


def standard_scale(mat, mean, std, reverse=False, eps=1e-10):
    if reverse:
        mat = mat * std + mean
    else:
        mat = (mat - mean) / std.clip(eps)
    return mat


def robust_scale(mat, q2, q3, q4, reverse=False, eps=1e-10):
    if reverse:
        mat = mat * (q4 - q2) + q3
    else:
        mat = (mat - q3) / (q4 - q2).clip(eps)
    return mat


def minmax_scale(mat, mi, ma, reverse=False, eps=1e-10):
    if reverse:
        mat = mat * (ma - mi) + mi
    else:
        mat = (mat - mi) / (ma - mi).clip(eps)
    return mat


def exp_scale(mat, lam, reverse=False, eps=1e-10):
    if reverse:
        mat = -np.log(1 - mat) / lam
    else:
        mat = 1 - np.exp(-lam * mat)
    return mat
