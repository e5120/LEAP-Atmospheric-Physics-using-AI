# python run/evaluate.py --experimental-rerun=/path/to/config.pickle
from pathlib import Path
from collections import defaultdict

import hydra
import torch
import polars as pl
import lightning as L
from tqdm.auto import tqdm

from leap import LeapDataModule, LeapModelModule
from leap.tfrecord import TFRecordDataLoader
from leap.utils import (
    setup,
    get_label_columns,
    IN_SCALAR_COLUMNS,
    IN_VECTOR_COLUMNS,
    OUT_SCALAR_COLUMNS,
    OUT_VECTOR_COLUMNS,
    OUT_COLUMNS,
)


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg):
    setup(cfg)
    val_loader = TFRecordDataLoader(Path(cfg.dir.data_dir, cfg.dataset_name), batch_size=cfg.batch_size, stage="val")
    test_data = defaultdict(list)
    for batch in tqdm(val_loader):
        test_data["sample_id"] += list(map(lambda x: x.decode(), batch["sample_id"].tolist()))
        for col in OUT_SCALAR_COLUMNS:
            test_data[col] += batch[col].tolist()
        for col in OUT_VECTOR_COLUMNS:
            for i in range(60):
                test_data[f"{col}_{i}"] += batch[col][:, i].tolist()
    test_df = pl.DataFrame(test_data)
    test_df = test_df.select(pl.col(pl.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"), n_rows=1).columns))
    test_df = test_df.select(pl.all().name.suffix("_label"))
    cfg.stage = "test"
    datamodule = LeapDataModule(cfg)
    datamodule.predict_dataloader = datamodule.val_dataloader
    cfg.model.params.input_size = len(IN_SCALAR_COLUMNS) + 60 * len(IN_VECTOR_COLUMNS)
    cfg.model.params.output_size = len(OUT_SCALAR_COLUMNS) + 60 * len(OUT_VECTOR_COLUMNS)
    label_columns = get_label_columns(OUT_COLUMNS)
    trainer = L.Trainer(**cfg.trainer)
    if cfg.dir.name == "kaggle":
        output_dir = Path("/kaggle/working")
        model_paths = Path(cfg.dir.model_dir, f"lb-{cfg.exp_name}").glob("*.ckpt")
    else:
        output_dir = Path(cfg.output_dir)
        model_paths = output_dir.glob("*.ckpt")
    for model_path in model_paths:
        print(model_path)
        modelmodule = LeapModelModule.load_from_checkpoint(
            checkpoint_path=model_path,
            cfg=cfg,
            label_columns=label_columns,
        )
        predictions = trainer.predict(modelmodule, datamodule=datamodule)
        predictions = torch.cat(predictions).double().numpy()
        assert predictions.shape[1] == len(label_columns), f"{predictions.shape}, {len(label_columns)}"
        submit_df = test_df.with_columns(
            [
                pl.lit(predictions[:, i]).alias(label_name)
                for i, label_name in enumerate(label_columns)
            ]
        )
        submit_df.write_csv(Path(output_dir, f"evaluation.csv"))


if __name__=="__main__":
    main()
