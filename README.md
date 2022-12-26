# DELG-pytorch
Pytorch Implementation of Unifying Deep Local and Global Features for Image Search ([delg-eccv20](https://arxiv.org/pdf/2001.05027.pdf))

## Installation

Install Python dependencies:

```
pip install -r requirements.txt
```

Set PYTHONPATH:

```
export PYTHONPATH=`pwd`:$PYTHONPATH
```

## Training

Training a delg model:

```
python train_delg.py \
    --cfg configs/metric/resnet_delg_8gpu.yaml \
    OUT_DIR ./output \
    PORT 12001 \
    TRAIN.WEIGHTS path/to/pretrainedmodel
```
Resume training: 

```
python train_delg.py \
    --cfg configs/metric/resnet_delg_8gpu.yaml \
    OUT_DIR ./output \
    PORT 12001 \
    TRAIN.AUTO_RESUME True
```

## Weights

-[r50-delg](https://pan.baidu.com/s/1rbB1ZItdMsyCiU5YrUbkCA) (wu46)

-[r101-delg](https://pan.baidu.com/s/1cahOcy9hx23RqHgBcKp1uQ)  (5pdj) 

pretrained weeights are available in [pymetric](https://github.com/feymanpriv/pymetric)


## Evaluation on Revisited Oxford & Paris

1. Install [**pydegensac**](https://github.com/ducha-aiki/pydegensac)

2. Extract global and local features with `extract_features.py`

3. Perform image retrieval with `perform_retrieval.py`


```
Using r50-delg pretrained model
- on roxford5k
1. With global features
    mAP E: 90.73, M: 77.3, H: 57.44
    mP@k[ 1  5 10] E: [95.59 92.35 91.07], M: [95.71 92.86 88.29], H: [90.   81.71 70.71]

2. With global and local features
    mAP E: 94.22, M: 78.18, H: 57.05
    mP@k[ 1  5 10] E: [100.    94.93  90.51], M: [98.57 96.38 93.24], H: [97.14 82.95 74.67]


- on rparis6k(updating)
1. With global features
    mAP E: 95.1, M: 88.12, H: 75.1
    mP@k[ 1  5 10] E: [95.71 97.14 95.79], M: [97.14 97.71 97.29], H: [95.71 92.86 92.43]

2. With global and local features
    mAP E: 96.03, M: 88.13, H: 74.13
    mP@k[ 1  5 10] E: [100.    97.43  96.36], M: [100.    99.14  98.29], H: [97.14 93.14 92.86]