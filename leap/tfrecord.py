import yaml
from pathlib import Path

import numpy as np
import polars as pl
import contextlib2
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm.auto import tqdm

from leap.utils import IN_SCALAR_COLUMNS, IN_VECTOR_COLUMNS, OUT_SCALAR_COLUMNS, OUT_VECTOR_COLUMNS


# https://www.kaggle.com/code/konstantinboyko/convert-original-csv-file-to-tfrecord/notebook
# https://www.kaggle.com/code/hidehisaarai1213/g2net-waveform-tfrecords
# https://www.kaggle.com/code/hidehisaarai1213/g2net-read-from-tfrecord-train-with-pytorch#TFRecord-Loader
def generate_tf_example(row, stage):
    feature = {"sample_id": string_feature(row[0, "sample_id"])}
    feature.update({
        col: float_feature(row[0, col])
        for col in IN_SCALAR_COLUMNS
    })
    feature.update({
        col: float_list_feature(row[0, col].to_numpy())
        for col in IN_VECTOR_COLUMNS
    })
    if stage != "test":
        feature.update({
            col: float_feature(row[0, col])
            for col in OUT_SCALAR_COLUMNS
        })
        feature.update({
            col: float_list_feature(row[0, col].to_numpy())
            for col in OUT_VECTOR_COLUMNS
        })
    features = tf.train.Features(feature=feature)
    proto_example = tf.train.Example(features=features)
    return proto_example


def get_tfrecords(exit_stack, data_dir, num_shards, stage):
    tfrecord_filenames = [
        str(Path(data_dir, f"{stage}_{idx:03d}.tfrecord"))
        for idx in range(num_shards)
    ]
    tfrecords = [
        exit_stack.enter_context(tf.io.TFRecordWriter(filename, options="GZIP"))
        for filename in tfrecord_filenames
    ]
    return tfrecords


def write_tfrecord(files, data_dir, stage, chunk_size=5, num_shards=100):
    shuffle = stage == "train"
    if shuffle:
        np.random.shuffle(files)
    n_data = 0
    gs = (len(files) - 1) // chunk_size + 1
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = get_tfrecords(tf_record_close_stack, data_dir, num_shards, stage)
        for i in range(gs):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            df = pl.concat([
                pl.read_parquet(filename)
                for filename in files[start: end]
            ])
            n_data += len(df)
            df = df.sample(n=len(df), shuffle=shuffle)
            for index in tqdm(range(len(df))):
                tf_example = generate_tf_example(df[index], stage)
                shard_index = index % num_shards
                output_tfrecords[shard_index].write(tf_example.SerializeToString())
    return n_data


def read_tfrecord(example, stage):
    tfrec_format = {"sample_id": tf.io.FixedLenFeature([], tf.string)}
    tfrec_format.update({
        col: tf.io.FixedLenFeature([], tf.float32)
        for col in IN_SCALAR_COLUMNS
    })
    tfrec_format.update({
        col: tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
        for col in IN_VECTOR_COLUMNS
    })
    if stage != "test":
        tfrec_format.update({
            col: tf.io.FixedLenFeature([], tf.float32)
            for col in OUT_SCALAR_COLUMNS
        })
        tfrec_format.update({
            col: tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True)
            for col in OUT_VECTOR_COLUMNS
        })
    example = tf.io.parse_single_example(example, tfrec_format)
    return example


def get_dataset(files, batch_size=1024, stage="train"):
    AUTO = tf.data.experimental.AUTOTUNE
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, compression_type="GZIP")
    # if stage in ["val", "test"]:
    #     ds = ds.cache()
    # ds = ds.repeat()
    if stage == "train":
        ds = ds.shuffle(1024 * 2)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
    ds = (
        ds
        .map(lambda example: read_tfrecord(example, stage), num_parallel_calls=AUTO)
        .batch(batch_size, drop_remainder=True if stage=="train" else False)
        .prefetch(AUTO)
    )
    return tfds.as_numpy(ds)


class TFRecordDataLoader(object):
    def __init__(self, data_dir, batch_size=1024, stage="train", num_files=None):
        assert stage in ["train", "val", "test"]
        with open(Path(data_dir, "data_size.yaml"), "r") as f:
            self.num_examples = yaml.safe_load(f)[stage]
        files = sorted(data_dir.glob(f"{stage}_*.tfrecord"))
        if num_files:
            self.num_examples = int(self.num_examples * num_files / len(files))
            files = np.random.choice(files, num_files, replace=False)
        files = list(map(str, files))
        self.ds = get_dataset(
            files,
            batch_size=batch_size,
            stage=stage,
        )
        self.batch_size = batch_size
        self.stage = stage
        self._iterator = None

    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        n_batches = (self.num_examples - 1) // self.batch_size
        if self.stage != "train":
            n_batches += 1
        return n_batches


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def string_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()]))


def string_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value.encode()))
