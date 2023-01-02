# Course project for Computer Vision
This is the course project for Computer Vision. We focus on improvement over the two-stage image retrieval. We incorporate supervised contrastive learning for global feature enhancement and dense keypoint matching for local feature verification. 

## Installation
```
conda env create -f environment.yaml
conda activate ImageRetrieval
```

## Training
Our implementation starts with open-sourced model from [DELG-ECCV20](https://arxiv.org/pdf/2001.05027.pdf) and [LoFTR-CVPR21](https://arxiv.org/pdf/2104.00680.pdf). 

### Trained model weights
1. [DELG models](https://pan.baidu.com/share/init?surl=rbB1ZItdMsyCiU5YrUbkCA), passwd wu46. 
2. [LoFTR models](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf)
3. [Our finetuned models](https://pan.baidu.com/s/13GrjXBDNg1ONEgiyOTmh_A), passwd ksfu. 

### Dataset download
The model is trained on Google Landmarks Dataset v2 ([GLDv2](https://github.com/cvdfoundation/google-landmark)). The train, index, and test sets are split into 500, 100, and 20 TAR files. The image files and labels can be downloaded using:
```
bash download_dataset.sh 500 100 20
```

### Global feature finetuning
Our global feature finetuning is based on the open-sourced DELG model: 
1. Put the pretrained DELG model at directory `logs`.
2. Put the downloaded GLDv2 data at directory `datasets/data/landmark`. 
3. For direct finetuning, run `train_delg.sh`. For supervised contrastive learning finetuning, run `train_supcon_delg.sh`. 


## Image Retrieval
### Benchmark download
The [Revisited Oxford & Paris](http://cmp.felk.cvut.cz/revisitop/) is a widely used benchmark for image retrieval. The Oxford subset has 70 queries and 5k index images; the Paris subset has 70 queries and 6k index images. We need the matlab ground truth files and image files, both of which can be found at the project page. 

### Retrieval process

Image retrieval includes two stages: 
1. Extract global and local features by running `extract_features.sh`. 
2. Perform retrieval with global ranking by running `perform_retrieval.sh`. 

To perform local feature extraction with LoFTR, download pretrained model weights and put to the directory `loftr_tools/weights`. 

We provide a group of [extracted features (passwd tc8j)](https://pan.baidu.com/s/1oNqyvatZ1w3aFiPQicvnZQ) for Oxford5k subset. 