import yaml
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from leap import LeapDataModule, LeapModelModule
from leap.utils import setup, get_num_training_steps, build_callbacks


@hydra.main(config_path="conf", config_name="train", version_base=None)
def main(cfg):
    setup(cfg)
    datamodule = LeapDataModule(cfg)
    cfg.model.params.input_size = datamodule.input_size
    cfg.model.params.output_size = datamodule.output_size
    with open(Path(cfg.dir.data_dir, "data_size.yaml"), "r") as f:
        num_train_data = yaml.safe_load(f)["train"]
        if cfg.num_train_files:
            files = list(Path(cfg.dir.data_dir).glob("train*.tfrecord"))
            num_train_data = int(num_train_data * cfg.num_train_files / len(files))
    max_steps = get_num_training_steps(num_train_data, cfg)
    if "num_training_steps" in cfg.scheduler.params:
        cfg.scheduler.params.num_training_steps = max_steps
    if "T_max" in cfg.scheduler.params:
        cfg.scheduler.params.T_max = max_steps
    if "total_steps" in cfg.scheduler.params:
        cfg.scheduler.params.total_steps = max_steps
    modelmodule = LeapModelModule(cfg)
    callbacks = build_callbacks(cfg)
    name = None if cfg.exp_name == "dummy" else cfg.exp_name
    logger = WandbLogger(project="leap", name=name) if cfg.logger else None
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **cfg.trainer,
    )
    trainer.fit(modelmodule, datamodule)


if __name__=="__main__":
    main()
