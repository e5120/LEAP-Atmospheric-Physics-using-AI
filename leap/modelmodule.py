import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch
import lightning as L
from torchmetrics import R2Score

import leap.model
import leap.optimizer
import leap.scheduler


class LeapModelModule(L.LightningModule):
    def __init__(self, label_columns, cfg):
        super().__init__()
        self.cfg = cfg
        self.label_columns = np.array(label_columns)
        self.output_dir = cfg.output_dir
        self.monitor = cfg.model_checkpoint.monitor
        if cfg.ignore_mask:
            sample_df = pl.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"), n_rows=1)
            tgt_cols = np.where(sample_df[0, 1:].to_numpy()[0] == 0, True, False)
            ignore_cols = np.array(sample_df.columns[1:])[tgt_cols]
            ignore_mask = []
            for col in label_columns:
                if col in ignore_cols:
                    ignore_mask.append(False)
                else:
                    ignore_mask.append(True)
            ignore_mask = torch.BoolTensor(ignore_mask)
        else:
            ignore_mask = None
        self.ignore_mask = ignore_mask
        self.model = getattr(leap.model, cfg.model.name)(ignore_mask=ignore_mask, **cfg.model.params)
        print(self.model)
        self.output_size = cfg.model.params.output_size
        self.metrics = R2Score(self.output_size, multioutput="raw_values")

    def forward(self, batch):
        return self.model(batch)

    def calculate_loss(self, batch, batch_idx):
        return self.model.calculate_loss(batch)

    def training_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        loss = ret["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ret = self.calculate_loss(batch, batch_idx)
        loss = ret["loss"]
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.metrics.update(ret["logits"], batch["labels"])

    def on_validation_epoch_end(self):
        raw_val_r2 = self.metrics.compute()
        self.metrics.reset()
        val_r2 = raw_val_r2.clone()
        if self.ignore_mask is not None:
            val_r2[~self.ignore_mask] = 1.0
        broken_mask = raw_val_r2 < 1e-6
        val_r2[broken_mask] = 0.0
        val_logs = {
            "val_r2": val_r2.mean(),
            "r2_raw": raw_val_r2.mean(),
            # "r2_std": val_r2.std(),
            "r2_broken": broken_mask.sum(),
        }
        self.log_dict(val_logs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # うまく学習できていないカラムを記録
        best_score = self.trainer.checkpoint_callback.best_model_score
        mode = self.trainer.checkpoint_callback.mode
        if best_score is None or \
           (mode == "max" and val_logs[self.monitor] >= best_score) or \
           (mode == "min" and val_logs[self.monitor] <= best_score):
            broken_label_columns = self.label_columns[broken_mask.detach().to("cpu").numpy()]
            with open(Path(self.output_dir, "broken_columns.pkl"), "wb") as f:
                pickle.dump(broken_label_columns, f)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        logits = self.forward(batch)["logits"]
        return logits

    def configure_optimizers(self):
        optimizer = getattr(leap.optimizer, self.cfg.optimizer.name)(
            self.parameters(),
            **self.cfg.optimizer.params,
        )
        scheduler = getattr(leap.scheduler, self.cfg.scheduler.name)(
            optimizer,
            **self.cfg.scheduler.params,
        )
        if self.cfg.scheduler.name in ["ReduceLROnPlateau"]:
            interval = "epoch"
        else:
            interval = "step"
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "monitor": "val_r2",
            }
        }
