# Attentive Information Bottleneck with Regularization

PyTorch code for "Information Bottleneck-Inspired Spatial Attention for Robust Semantic Segmentation in Autonomous Driving (ICARCV2024)".

## Environments
- Python 3.8.19
- torch 1.13.1+cu117
- torchvision 0.14.1+cu117
- mmcv 2.1.0
- mmsegmentation 1.2.2


## Datasets Preparation

The datasets are arranged as follows:
```
data
 |---bdd100k
      |---images/10k
           |---test
           |---train
           |---val
      |---labels/sem_seg
           |---masks
                |---train
                |---val
 |---cityscapes
      |---gtFine
           |---test
           |---train
                |---aachen/
                     |---color.png, instanceIds.png, labelIds.png, polygons.json, labelTrainIds.png
                |--- ...
           |---val
      |---leftImg8bit
           |---test
           |---train
           |---val
 |---nightcity-fine      
      |---train
           |---img
           |---lbl
      |---val
           |---img
           |---lbl
```

### BDD100K
Download from [Official Website](https://www.mapillary.com/dataset/vistas).

### Cityscapes
Download from [Official Website](https://www.cityscapes-dataset.com/downloads/).

To create `labelTrainIds.png`, first install cityscapes scripts by:
```
pip install git+https://github.com/mcordts/cityscapesScripts.git
```

Then run `cityscapesescript` with:
```
CITYSCAPES_DATASET=/path/to/abovementioned/cityscapes python cityscapesscripts/preparation/createTrainIdLabelImgs.py
```

### NightCity-fine
Download NightCity-fine from [Google Drive](https://drive.google.com/file/d/1Ilj99NMAmkZIPQcVOd6cJebnKXjJ-Sit/view?usp=drive_link). 



## Domain-generalized Semantic Segmentation

Models are trained on Cityscapes and tested on BDD100K.

For training ResNet-50 backbone:
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r50_aib_v4-d8_4xb2-40k_cityscapes-512x1024_vabeta0.0001.py
PORT=29500 bash tools/dist_train.sh ${CONFIG} ${GPUS}
```

For testing ResNet-50 backbone on BDD100K: 
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r50_aib-d8_4xb2-40k_cityscapes_bdd100k-512x1024.py
CKPT=work_dirs/deeplabv3plus_r50_aib_v4-d8_4xb2-40k_cityscapes-512x1024_vabeta0.0001/iter_40000.pth
PORT=29501 bash tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS}
```

For training ResNet-101 backbone:
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r101_aib_v4-d8_4xb2-40k_cityscapes-512x1024.py
PORT=29500 bash tools/dist_train.sh ${CONFIG} ${GPUS}
```

For testing ResNet-101 backbone on BDD100K: 
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r101_aib-d8_4xb2-40k_cityscapes_bdd100k-512x1024.py
CKPT=work_dirs/deeplabv3plus_r101_aib-d8_4xb2-40k_cityscapes-512x1024_vabeta0.001/iter_40000.pth
PORT=29501 bash tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS}
```

## Nighttime Semantic Segmentation
Models are trained on Cityscapes (denoted as $\mathcal{C}$) and NightCity-fine (denoted as $\mathcal{N}$-fine), and then tested on  $\mathcal{C}$ or $\mathcal{N}$-fine.


For training ResNet-50 backbone:
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r50_aib_v4-d8_4xb2-80k_cityscapes_nightcity-512x1024.py
PORT=29500 bash tools/dist_train.sh ${CONFIG} ${GPUS}
```

This will give evaluation results on NighCity-fine at the end of the training.

For testing ResNet-50 backbone on Cityscapes: 
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r50_aib_v4-d8_4xb2-40k_cityscapes-512x1024.py
CKPT=work_dirs/deeplabv3plus_r50_aib_v4-d8_4xb2-80k_cityscapes_nightcity-512x1024/iter_80000.pth
PORT=29501 bash tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS}
```

For training ResNet-101 backbone:
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r101_aib_v4-d8_4xb2-80k_cityscapes_nightcity-512x1024.py
PORT=29500 bash tools/dist_train.sh ${CONFIG} ${GPUS}
```

This will give evaluation results on NighCity-fine at the end of the training.


For testing ResNet-101 backbone on Cityscapes: 
```Bash
GPUS=4
CONFIG=configs/deeplabv3plus_aibr/deeplabv3plus_r101_aib_v4-d8_4xb2-80k_cityscapes-512x1024.py
CKPT=work_dirs/deeplabv3plus_r101_aib_v4-d8_4xb2-80k_cityscapes_nightcity-512x1024/iter_80000.pth
PORT=29501 bash tools/dist_test.sh ${CONFIG} ${CKPT} ${GPUS}
```

## Model Weights Download

Pre-trained models and visualized maps can be downloaded from [BaiduDisk](https://pan.baidu.com/s/1KUqYZZelZhPwIiCVNW9OFA?pwd=4iit). You may download the model weights, put them into `work_dirs`, and run the test commands above.

## Related Repositories
- [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
- [HGFormer](https://github.com/dingjiansw101/HGFormer/tree/main)
- [DTP](https://github.com/w1oves/DTP)
