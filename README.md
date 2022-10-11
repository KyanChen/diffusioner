

## CD-Diffuser

Here, we provide the pytorch implementation of the paper: XXX.

## log

- 尝试1（X）：V2-0703
  UNet结构
  AB做孪生encoder，
  mask仅在decoder中出现，与AB左营尺度做concat

## Training

```
python scripts/cd_res_train.py
```

## Evaluation

### 1. Sampling

采样脚本：`scripts/run_sample.sh`

```
python scripts/cd_res_sample.py
```

目标是：根据训练好的模型，通过已有的双时相图像，获得预测的标签；并落盘；

### 2. evaluation

目标：把落盘的预测mask与GT做比较，得到准确率等指标；

## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

### Data Download 

LEVIR-CD: https://justchenhao.github.io/LEVIR/

WHU-CD: https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html

GZ-CD: https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery

## License

Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Thanks

guided-diffusion:  https://github.com/openai/guided-diffusion
