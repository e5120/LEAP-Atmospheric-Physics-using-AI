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
$ export DATA_DIR=/path/to/data
$ python run/prepare_data.py phase=split data_dir=$DATA_DIR  # It takes about 1 hour
$ cp resources/* $DATA_DIR
$ python run/prepare_data.py phase=generate data_dir=$DATA_DIR # It takes about 1 hour
$ python run/prepare_data.py phase=tfrecord data_dir=$DATA_DIR # It takes about 2 hours
```


## Feedforward Network

- CV :
- public LB : 0.60847
- private LB : 

```
$ python run/train.py model=ffn_model exp_name=exp_ffn
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_ffn/single/.hydra/config.pickle
```

## CNN Model

- CV : 0.6901
- public LB : 0.73015
- private LB : 

```
$ python run/train.py model=light_cnn_model exp_name=exp_cnn model.params.out_channels=256
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_cnn/single/.hydra/config.pickle
```

## Resnet Model

- CV : 
- public LB : 
- private LB : 

```
$ python run/train.py model=cnn_model exp_name=exp_resnet
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_resnet/single/.hydra/config.pickle
```

## Transformer Encoder

- CV : 0.6395
- public LB : 0.65343
- private LB : 

```
$ python run/train.py model=attn_model exp_name=exp_attn
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_attn/single/.hydra/config.pickle
```
