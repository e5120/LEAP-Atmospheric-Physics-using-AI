from pathlib import Path

import hydra
import polars as pl
from tqdm.auto import tqdm

from leap.utils import NUM_TRAIN, IN_SCALAR_COLUMNS, IN_VECTOR_COLUMNS, OUT_SCALAR_COLUMNS, OUT_VECTOR_COLUMNS


def split_dataset(cfg):
    data_dir = Path(cfg.data_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("splitting dataset")
    weight_df = pl.read_csv(Path(data_dir, "sample_submission.csv"), n_rows=1)
    weight = weight_df[:, 1:].to_pandas().T.to_dict()[0]
    sample_df = pl.read_csv(Path(data_dir, "train.csv"), n_rows=100_000)
    gs = (NUM_TRAIN - 1) // cfg.batch_size + 1
    for i in tqdm(range(gs)):
        start = i * cfg.batch_size
        end = (i + 1) * cfg.batch_size - 1
        df = pl.read_csv(
            Path(data_dir, "train.csv"),
            n_rows=cfg.batch_size,
            skip_rows=start,
            schema=sample_df.schema,
        )
        df = df.with_columns(
            [
                pl.col(col) * weight[col]
                for col in weight_df.columns[1:]
            ]
        )
        df.write_parquet(Path(output_dir, f"{cfg.prefix}train_{start}_{end}.parquet"))
    test_df = pl.read_csv(Path(data_dir, "test.csv"))
    test_df.write_parquet(Path(output_dir, f"{cfg.prefix}test.parquet"))


def generate_sequential_data(cfg):
    data_dir = Path(cfg.data_dir)
    print("generating dataset")
    # train data
    for filename in tqdm(sorted(list(data_dir.glob(f"{cfg.prefix}train*")))):
        df = pl.read_parquet(filename)
        for col in IN_VECTOR_COLUMNS + OUT_VECTOR_COLUMNS:
            df = df.with_columns(pl.concat_list(f"^{col}_\d+$").alias(col))
        df = df.select(["sample_id"] + IN_SCALAR_COLUMNS + IN_VECTOR_COLUMNS + OUT_SCALAR_COLUMNS + OUT_VECTOR_COLUMNS)
        df.write_parquet(filename)
    # test data
    filename = Path(data_dir, f"{cfg.prefix}test.parquet")
    test_df = pl.read_parquet(filename)
    for col in IN_VECTOR_COLUMNS:
        test_df = test_df.with_columns(pl.concat_list(f"^{col}_\d+$").alias(col))
    test_df = test_df.select(["sample_id"] + IN_SCALAR_COLUMNS + IN_VECTOR_COLUMNS)
    test_df.write_parquet(filename)


@hydra.main(config_path="conf", config_name="prepare_data", version_base=None)
def main(cfg):
    if cfg.output_dir is None:
        cfg.output_dir = cfg.data_dir
    if cfg.phase == "split":
        split_dataset(cfg)
    if cfg.phase == "sequential":
        generate_sequential_data(cfg)


if __name__=="__main__":
    main()
