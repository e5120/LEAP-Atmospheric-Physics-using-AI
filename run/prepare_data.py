from pathlib import Path

import hydra
import polars as pl
import pandas as pd
import pyarrow as pa
from pyarrow import csv
from tqdm.auto import tqdm

from leap.utils import (
    NUM_TRAIN,
    IN_SCALAR_COLUMNS,
    IN_VECTOR_COLUMNS,
    OUT_SCALAR_COLUMNS,
    OUT_VECTOR_COLUMNS,
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


def generate_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    if not cfg.overwrite and list(data_dir.glob(f"{cfg.prefix}train*")):
        print("already exist")
        return
    print("generating dataset")
    label_cols = pl.read_csv(Path(data_dir, "sample_submission.csv"), n_rows=1).columns[1:]
    feat_cols = pl.read_csv(Path(data_dir, "train.csv"), n_rows=1).select(pl.exclude(label_cols)).columns[1:]
    # train data
    for filename in tqdm(sorted(list(data_dir.glob("raw_train*")))):
        df = pl.read_parquet(filename)
        df = normalize(df, feat_cols, label_cols, cfg.scaler, data_dir)
        df = df.with_columns(pl.col(pl.Float64).cast(pl.Float32))
        for col in IN_VECTOR_COLUMNS + OUT_VECTOR_COLUMNS:
            df = df.with_columns(pl.concat_list(f"^{col}_\d+$").alias(col))
        df = df.select(["sample_id"] + IN_SCALAR_COLUMNS + IN_VECTOR_COLUMNS + OUT_SCALAR_COLUMNS + OUT_VECTOR_COLUMNS)
        df.write_parquet(str(filename).replace("raw_", cfg.prefix))
    # test data
    filename = Path(data_dir, "raw_test.parquet")
    test_df = pl.read_parquet(filename)
    test_df = normalize(test_df, feat_cols, [], cfg.scaler, data_dir)
    for col in IN_VECTOR_COLUMNS:
        test_df = test_df.with_columns(pl.concat_list(f"^{col}_\d+$").alias(col))
    test_df = test_df.select(["sample_id"] + IN_SCALAR_COLUMNS + IN_VECTOR_COLUMNS)
    test_df.write_parquet(str(filename).replace("raw_", cfg.prefix))


@hydra.main(config_path="conf", config_name="prepare_data", version_base=None)
def main(cfg):
    if cfg.output_dir is None:
        cfg.output_dir = cfg.data_dir
    if cfg.phase == "split":
        split_dataset(cfg)
    if cfg.phase == "generate":
        generate_dataset(cfg)


if __name__=="__main__":
    main()
