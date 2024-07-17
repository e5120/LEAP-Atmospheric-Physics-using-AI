# [LEAP-Atmospheric-Physics-using-AI](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/overview)

## Result
- 31 / 693
- public LB : 0.77646
- private LB : 0.77090
    - I attempted to use data leak information, but I did not utilize it in the final submission.

| model | CV | public LB | private LB |
|:--|--:|--:|--:|
|[FFN](#feedforward-network-ffn)|0.6177|0.62068|0.61846|
|[FFN v2](#feedforward-network-v2-ffn-v2)|0.6334|0.63881|0.63622|
|[Transformer](#transformer)|0.7304|0.72976|0.72350|
|[Resnet](#resnet)|0.7517|0.75101|0.74707|
|[UNet](#unet)|0.7353|0.73291|0.72949|
|[UNet w/ squeeze-and-excitation](#unet-with-squeeze-and-excitation)|0.7556|0.75313|0.74363|
|[LightCNN](#lightcnn-model)|0.7377|0.73778|0.73312|
|[LightCNN v2](#lightcnn-model-v2)|0.7638|0.76333|0.75920|
|[LightCNN v3](#lightcnn-model-v3)|0.7546|0.75262|0.74656|
|[SqueezeFormer](#squeezeformer-model)|0.7522|0.75250|0.74571|
|[ASLFR](#aslfr-model)|0.7645|0.76416|0.76089|

## Computational Resouces
- Intel(R) Core(TM) i5-8400 CPU @ 2.80GHz
- 64GB RAM
- 2x NVIDIA GeForce GTX 1080Ti

## Download Data

```
$ kaggle competitions download -c leap-atmospheric-physics-ai-climsim
$ kaggle datasets download -d rkwasny/pbuf-ozone-2-mapping-csv  # If you use data leakage
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
$ export DATASET=dataset_v1
$ python run/prepare_data.py phase=generate data_dir=$DATA_DIR dataset_name=$DATASET # It takes about 1 hour
$ python run/prepare_data.py phase=tfrecord data_dir=$DATA_DIR dataset_name=$DATASET # It takes about 4 hours
```


## Training & Inference

### Feedforward Network (FFN)

```
$ python run/train.py dataset_name=$DATASET model=ffn_model exp_name=exp_ffn
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_ffn/single/.hydra/config.pickle
```

### Feedforward Network v2 (FFN v2)

```
$ python run/train.py dataset_name=$DATASET model=ffn_model_v2 exp_name=exp_ffn_v2
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_ffn_v2/single/.hydra/config.pickle
```

### Transformer

```
$ python run/train.py dataset_name=$DATASET model=attn_model exp_name=exp_attn
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_attn/single/.hydra/config.pickle
```

### Resnet

```
$ python run/train.py dataset_name=$DATASET model=resnet_model exp_name=exp_resnet
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_resnet/single/.hydra/config.pickle
```

### UNet

```
$ python run/train.py dataset_name=$DATASET model=unet_model exp_name=exp_unet
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_unet/single/.hydra/config.pickle
```

### UNet with Squeeze-and-Excitation

```
$ python run/train.py dataset_name=$DATASET model=unet_with_se_model exp_name=exp_unet_with_se
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_unet_with_se/single/.hydra/config.pickle
```

### Resnet Model

```
$ python run/train.py dataset_name=$DATASET model=cnn_model exp_name=exp_resnet
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_resnet/single/.hydra/config.pickle
```

### LightCNN Model

```
$ python run/train.py dataset_name=$DATASET model=light_cnn_model exp_name=exp_light_cnn
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_light_cnn/single/.hydra/config.pickle
```

### LightCNN Model v2

```
$ python run/train.py dataset_name=$DATASET model=light_cnn_model_v2 exp_name=exp_light_cnn_v2
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_light_cnn_v2/single/.hydra/config.pickle
```

### LightCNN Model v3

```
$ python run/train.py dataset_name=$DATASET model=light_cnn_model_v3 exp_name=exp_light_cnn_v3
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_light_cnn_v3/single/.hydra/config.pickle
```

### SqueezeFormer Model

```
$ python run/train.py dataset_name=$DATASET model=squeeze_former_model exp_name=exp_squeeze_former
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_squeeze_former/single/.hydra/config.pickle
```

### ASLFR Model

```
$ python run/train.py dataset_name=$DATASET model=aslfr_model exp_name=exp_aslfr
$ python run/inference.py --experimental-rerun=/path/to/repo/output/exp_aslfr/single/.hydra/config.pickle
```
