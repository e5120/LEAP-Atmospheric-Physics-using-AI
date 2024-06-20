# python run/inference.py --experimental-rerun=/path/to/config.pickle
import pickle
from pathlib import Path

import hydra
import torch
import polars as pl
import lightning as L

from leap import LeapDataModule, LeapModelModule
from leap.utils import setup, normalize, get_label_columns


def post_process(df, cfg):
    df = normalize(df, [], df.columns[1:], "standard", cfg.dir.data_dir, reverse=True)
    # うまく学習できていないカラムを平均値で置き換え
    with open(Path(cfg.output_dir, "broken_columns.pkl"), "rb") as f:
        broken_label_columns = pickle.load(f)
    stats_df = pl.read_parquet(Path(cfg.dir.data_dir, "label_stats.parquet"))
    for col in broken_label_columns:
        df = df.with_columns(pl.lit(stats_df[0, col]).alias(col))
    # clipping (0以上の値しか取らないのに，負の値を予測した場合に0にする)
    df = df.with_columns(
        pl.col(["cam_out_NETSW", "cam_out_PRECSC", "cam_out_PRECC", "cam_out_SOLS", "cam_out_SOLL", "cam_out_SOLSD", "cam_out_SOLLD"]).clip(lower_bound=0.0)
    )
    # 入力から陽に計算できるものはモデルの出力値を使わない
    # https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/discussion/502484
    n = 30
    columns = [f"state_q0002_{i}" for i in range(n)]
    input_df = pl.read_csv(Path(cfg.dir.data_dir, "test.csv"), columns=columns)
    for i in range(n):
        df = df.with_columns(
            pl.lit(-input_df[f"state_q0002_{i}"] / 1200).alias(f"ptend_q0002_{i}")
        )
    # 重みをかける
    weight_df = pl.read_csv(Path(cfg.dir.data_dir, "sample_submission.csv"), n_rows=1)[:, 1:]
    columns = weight_df.columns
    for col in columns:
        assert col in df
        df = df.with_columns(
            pl.col(col) * weight_df[0, col]
        )
    # 提出用の並び順にする
    df = df.select(pl.col(["sample_id"] + columns))
    return df


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(cfg):
    setup(cfg)
    cfg.stage = "test"
    datamodule = LeapDataModule(cfg)
    cfg.model.params.input_size = datamodule.input_size
    cfg.model.params.output_size = datamodule.output_size
    test_df = pl.read_parquet(Path(cfg.dir.data_dir, "processed_test.parquet"), columns=["sample_id"])
    label_columns = get_label_columns(datamodule.label_columns)
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
        submit_df = post_process(submit_df, cfg)
        submit_df.write_csv(Path(output_dir, f"submission_{model_path.stem}.csv"))


if __name__=="__main__":
    main()
