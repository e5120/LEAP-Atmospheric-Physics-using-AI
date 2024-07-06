import yaml
from pathlib import Path

import hydra
import polars as pl
import pandas as pd
import pyarrow as pa
from pyarrow import csv
from tqdm.auto import tqdm

from leap.tfrecord import write_tfrecord
from leap.utils import (
    IN_SCALAR_COLUMNS,
    IN_VECTOR_COLUMNS,
    OUT_SCALAR_COLUMNS,
    OUT_VECTOR_COLUMNS,
    NUM_GRID,
    normalize,
)


def split_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if list(data_dir.glob("raw_train*")):
        print("already exist")
        return
    print("splitting dataset")
    reader = pa.csv.open_csv(Path(data_dir, "train.csv"))
    n_data = 0
    start = n_data
    dfs = []
    for batch in tqdm(reader):
        n_data += len(batch)
        df = pa.Table.from_batches([batch]).to_pandas()
        dfs.append(df)
        if n_data >= cfg.batch_size:
            df = pd.concat(dfs).reset_index(drop=True)
            end = start + n_data - 1
            df.to_parquet(Path(output_dir, f"raw_train_{start}_{end}.parquet"))
            start += n_data
            n_data = 0
            dfs = []
    test_df = pl.read_csv(Path(data_dir, "test.csv"))
    test_df.write_parquet(Path(output_dir, "raw_test.parquet"))


def feature_engineering(df, is_test):
    if is_test:
        df = df.with_columns(
            pl.lit(-1).alias("location"),
            pl.lit(-1).alias("timestamp"),
        )
    else:
        df = df.with_columns(
            # (pl.col("sample_id").str.extract(r"(\d+)").str.to_integer() // 1261440 / 8).alias("time")
            (pl.col("sample_id").str.extract(r"(\d+)").str.to_integer() % NUM_GRID).alias("location"),
            (pl.col("sample_id").str.extract(r"(\d+)").str.to_integer() // NUM_GRID).alias("timestamp"),
        )
    add_feats = ["location", "timestamp"]
    return df, add_feats


def generate_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir, cfg.dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.overwrite and list(output_dir.glob(f"{cfg.prefix}*.parquet")):
        print("already exist")
        return
    print("generating dataset")
    scaler_method = {}
    for k, v in cfg.scaler.items():
        scaler_method[k] = list(v)
    with open(Path(output_dir, "scaler_methods.yaml"), "w") as f:
        yaml.safe_dump(scaler_method, f)
    # train data
    for filename in tqdm(sorted(list(data_dir.glob("raw_train*")))):
        df = pl.read_parquet(filename)
        df, add_feats = feature_engineering(df, False)
        df = normalize(df, scaler_method, data_dir)
        df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        for col in IN_VECTOR_COLUMNS + OUT_VECTOR_COLUMNS:
            df = df.with_columns(pl.concat_list(f"^{col}_\d+$").alias(col))
        df = df.select(["sample_id"] + IN_SCALAR_COLUMNS + IN_VECTOR_COLUMNS + OUT_SCALAR_COLUMNS + OUT_VECTOR_COLUMNS + add_feats)
        save_filename = filename.stem.replace("raw_", cfg.prefix) + ".parquet"
        df.write_parquet(Path(output_dir, save_filename))
    # test data
    filename = Path(data_dir, "raw_test.parquet")
    test_df = pl.read_parquet(filename)
    test_df, add_feats = feature_engineering(test_df, True)
    test_df = normalize(test_df, scaler_method, data_dir)
    for col in IN_VECTOR_COLUMNS:
        test_df = test_df.with_columns(pl.concat_list(f"^{col}_\d+$").alias(col))
    test_df = test_df.select(["sample_id"] + IN_SCALAR_COLUMNS + IN_VECTOR_COLUMNS + add_feats)
    save_filename = filename.stem.replace("raw_", cfg.prefix) + ".parquet"
    test_df.write_parquet(Path(output_dir, save_filename))


def generate_tfrecord(cfg):
    data_dir = Path(cfg.output_dir, cfg.dataset_name)
    if not cfg.overwrite and list(data_dir.glob(f"*.tfrecord")):
        print("already exist")
        return
    print("generating tfrecord")
    files = sorted(data_dir.glob(f"{cfg.prefix}train*.parquet"))
    trn_files = files[:-cfg.num_val_files]
    val_files = files[-cfg.num_val_files:]
    test_files = sorted(data_dir.glob(f"{cfg.prefix}test.parquet"))
    print(len(trn_files), len(val_files), len(test_files))
    num_train_data = write_tfrecord(trn_files, data_dir, "train", chunk_size=cfg.chunk_size, num_shards=cfg.num_shards)
    num_val_data = write_tfrecord(val_files, data_dir, "val", num_shards=5)
    num_test_data = write_tfrecord(test_files, data_dir, "test", num_shards=1)
    with open(Path(data_dir, "data_size.yaml"), "w") as f:
        num_data = {
            "train": num_train_data,
            "val": num_val_data,
            "test": num_test_data,
        }
        yaml.safe_dump(num_data, f)


@hydra.main(config_path="conf", config_name="prepare_data", version_base=None)
def main(cfg):
    if cfg.output_dir is None:
        cfg.output_dir = cfg.data_dir
    if cfg.phase == "split":
        split_dataset(cfg)
    if cfg.phase == "generate":
        generate_dataset(cfg)
    if cfg.phase == "tfrecord":
        generate_tfrecord(cfg)


if __name__=="__main__":
    main()
