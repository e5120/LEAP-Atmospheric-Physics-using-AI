# [LEAP-Atmospheric-Physics-using-AI](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview)

## Computational Resouces
- Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
- 64GB RAM
- 2x NVIDIA GeForce GTX 1080Ti

## Download Data

```
$ kaggle competitions download -c leap-atmospheric-physics-ai-climsim
```

## Download this repository

```
$ git clone https://github.com/e5120/LEAP-Atmospheric-Physics-using-AI.git
$ cd LEAP-Atmospheric-Physics-using-AI
$ export PYTHONPATH=.
```

## Prepare Data

```
$ python run/prepare_data.py phase=split data_dir=/path/to/data  # It takes about 1 hour
$ cp resources/* /path/to/data
$ python run/prepare_data.py phase=generate data_dir=/path/to/data # It takes about 1 hour
$ python run/prepare_data.py phase=tfrecord data_dir=/path/to/data # It takes about 2 hours
```

## Train

```
$ python run/train.py dataset=single_dataset model=ffn_model exp_name=exp001
```

## Inference

```
$ python run/inference.py --config-path=/path/to/repo/output/exp001/single/.hydra
```
