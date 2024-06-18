import yaml
import pickle
from pathlib import Path

import numpy as np
import hydra
import lightning as L
from lightning.pytorch.loggers import WandbLogger

from leap import LeapDataModule, LeapModelModule
from leap.utils import setup, get_num_training_steps, build_callbacks, get_label_columns


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
    label_columns = get_label_columns(datamodule.label_columns)
    modelmodule = LeapModelModule(label_columns, cfg)
    callbacks = build_callbacks(cfg)
    name = None if cfg.exp_name == "dummy" else cfg.exp_name
    logger = WandbLogger(project="leap", name=name) if cfg.logger else None
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **cfg.trainer,
    )
    trainer.fit(modelmodule, datamodule)
    # 後処理 (うまく学習できていないカラムを記録)
    broken_mask = modelmodule.broken_mask
    broken_label_columns = np.array(label_columns)[~broken_mask]
    print(broken_label_columns)
    print(f"# of broken columns: {len(broken_label_columns)}")
    with open(Path(cfg.dir.model_dir, cfg.exp_name, cfg.dir_name, "broken_columns.yaml"), "wb") as f:
        pickle.dump(broken_label_columns, f)


if __name__=="__main__":
    main()
