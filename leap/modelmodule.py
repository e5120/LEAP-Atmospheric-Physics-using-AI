from pathlib import Path

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
        if cfg.ignore_mask:
            sample_df = pl.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"), n_rows=1)
            ignore_cols = sample_df.select(pl.col(pl.Int64)).columns
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
        self.broken_mask = None

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
        val_r2 = self.metrics.compute()
        if self.ignore_mask is not None:
            val_r2[~self.ignore_mask] = 1.0
        broken_mask = val_r2 > 1e-6
        val_r2_sub = val_r2[broken_mask]
        self.broken_mask = broken_mask.detach().to("cpu").numpy()
        val_logs = {
            "val_r2": val_r2_sub.mean(),
            "val_r2_clipped": val_r2_sub.sum() / self.output_size,
            "val_r2_all": val_r2.mean(),
            # "val_r2_std": val_r2_sub.std(),
            "r2_broken": len(val_r2) - len(val_r2_sub),
        }
        self.log_dict(val_logs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.metrics.reset()

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
