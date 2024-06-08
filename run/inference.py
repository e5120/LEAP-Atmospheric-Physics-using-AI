from pathlib import Path

import hydra
import torch
import polars as pl
import lightning as L

from leap import LeapDataModule, LeapModelModule
from leap.utils import normalize


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg):
    cfg.stage = "test"
    datamodule = LeapDataModule(cfg)
    cfg.model.params.input_size = len(datamodule.feature_columns)
    cfg.model.params.output_size = len(datamodule.label_columns)
    test_dataloader = datamodule.test_dataloader()
    label_columns = datamodule.label_columns
    test_df = test_dataloader.dataset.df.select(["sample_id"])
    trainer = L.Trainer(**cfg.trainer)
    if cfg.dir.name == "kaggle":
        model_paths = Path(cfg.dir.model_dir, f"lb-{cfg.exp_name}").glob("*.ckpt")
        output_dir = Path("/kaggle/working")
    else:
        model_paths = Path(cfg.dir.model_dir, cfg.exp_name, cfg.dir_name).glob("*.ckpt")
        output_dir = Path(cfg.dir.model_dir, cfg.exp_name, cfg.dir_name)
    for model_path in model_paths:
        print(model_path)
        modelmodule = LeapModelModule.load_from_checkpoint(
            checkpoint_path=model_path,
            cfg=cfg,
        )
        predictions = trainer.predict(modelmodule, test_dataloader)
        predictions = torch.cat(predictions).numpy()
        submit_df = test_df.with_columns(
            [
                pl.lit(predictions[:, i]).alias(label_name)
                for i, label_name in enumerate(label_columns)
            ]
        )
        submit_df = normalize(submit_df, None, label_columns, cfg.scaler, cfg.dir.data_dir, reverse=True)
        columns = pl.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"), n_rows=1).columns
        submit_df = submit_df.select(pl.col(columns))
        submit_df.write_csv(Path(output_dir, f"submission_{model_path.stem}.csv"))


if __name__=="__main__":
    main()
