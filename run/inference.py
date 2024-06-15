from pathlib import Path

import hydra
import torch
import polars as pl
import lightning as L

from leap import LeapDataModule, LeapModelModule
from leap.utils import normalize, get_label_columns


# REPLACE_FROM = ['ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3', 'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7', 'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0002_15', 'ptend_q0002_16', 'ptend_q0002_17', 'ptend_q0002_18', 'ptend_q0002_19', 'ptend_q0002_20', 'ptend_q0002_21', 'ptend_q0002_22', 'ptend_q0002_23', 'ptend_q0002_24', 'ptend_q0002_25', 'ptend_q0002_26']
# REPLACE_TO = ['state_q0002_0', 'state_q0002_1', 'state_q0002_2', 'state_q0002_3', 'state_q0002_4', 'state_q0002_5', 'state_q0002_6', 'state_q0002_7', 'state_q0002_8', 'state_q0002_9', 'state_q0002_10', 'state_q0002_11', 'state_q0002_12', 'state_q0002_13', 'state_q0002_14', 'state_q0002_15', 'state_q0002_16', 'state_q0002_17', 'state_q0002_18', 'state_q0002_19', 'state_q0002_20', 'state_q0002_21', 'state_q0002_22', 'state_q0002_23', 'state_q0002_24', 'state_q0002_25', 'state_q0002_26']


def post_process(df, label_columns, cfg):
    df = normalize(df, [], label_columns, "standard", cfg.dir.data_dir, reverse=True)
    # https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484
    if False:
        columns = [f"state_q0002_{i}" for i in range(30)]
        input_df = pl.read_csv(Path(cfg.dir.data_dir, "test.csv"), columns=columns)
        for i in range(30):
            df.with_columns(
                pl.lit(-input_df[f"state_q0002_{i}"] / 1200).alias(f"ptend_q0002_{i}")
            )
    # 重みをかける
    weight_df = pl.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"), n_rows=1)[:, 1:]
    columns = weight_df.columns
    for col in columns:
        if col in df:
            df = df.with_columns(
                pl.col(col) * weight_df[0, col]
            )
        else:
            print(col)
            df = df.with_columns(pl.lit(0).alias(col))
    # 提出用の並び順にする
    df = df.select(pl.col(["sample_id"] + columns))
    return df


@hydra.main(config_path=None, config_name="config", version_base=None)
def main(cfg):
    cfg.stage = "test"
    datamodule = LeapDataModule(cfg)
    cfg.model.params.input_size = datamodule.input_size
    cfg.model.params.output_size = datamodule.output_size
    test_df = pl.read_parquet(Path(cfg.dir.data_dir, "processed_test.parquet"), columns=["sample_id"])
    label_columns = get_label_columns(datamodule.label_columns)
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
        predictions = trainer.predict(modelmodule, datamodule=datamodule)
        predictions = torch.cat(predictions).double().numpy()
        assert predictions.shape[1] == len(label_columns), f"{predictions.shape}, {len(label_columns)}"
        submit_df = test_df.with_columns(
            [
                pl.lit(predictions[:, i]).alias(label_name)
                for i, label_name in enumerate(label_columns)
            ]
        )
        submit_df = post_process(submit_df, label_columns, cfg)
        submit_df.write_csv(Path(output_dir, f"submission_{model_path.stem}.csv"))


if __name__=="__main__":
    main()
