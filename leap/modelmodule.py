import lightning as L
from torchmetrics import R2Score

import leap.model
import leap.optimizer
import leap.scheduler


class LeapModelModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = getattr(leap.model, cfg.model.name)(**cfg.model.params)
        self.metrics = R2Score(cfg.model.params.output_size)

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
        self.metrics.reset()
        self.log("val_r2", val_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

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
