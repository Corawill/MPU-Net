# MPU-Net
_Grain Boundary Detection in Multi-phase Material Microscopic Image via Deep Learning and Rectify Strategies_

The the PyTorch implementation of our paper MPU-Net.

## Abstract
In material science, image segmentation is of great significance for the quantitative analysis of microstructures, which is a crucial step in building the processing-structure-properties relationship. Although deep learning has become the mainstream in the field of material microscopic image analysis, the complexity of the microstructure still creates challenges to the boundary detection task. For example, the precipitate always covers the grain boundary, resulting in the over-segmentation of grains and hindering the accurate characterization of microstructure. In this work, we propose a novel method Multi-phase U-Net, dubbed as MPU-Net, to effectively detect the grain boundaries and precipitate at the same time in an end-to-end manner. In addition, we design two post-processing strategies to further improve boundary detection results. The adaptive grain redistribution strategy can rectify the over-segmentation errors of grain boundary caused by the cover of the precipitate phase. The pruning strategy can eliminate the non-closure boundary and improve the clarity of visualization results. Extensive experiment results on two classical datasets demonstrate that the proposed method achieves promising performance in both objective and subjective assessment.

![alt text](/show/framwork.png "overview")

## Environment
```
pip install -r requirements.txt
```
gala is installed according to https://github.com/janelia-flyem/gala.

## Quick Start
```
# inference
python main.py

# train MPU-Net
python train.py
```
Pre-train parameters download:
For wpunet segmentation, you can download at [Baidu Pan](https://pan.baidu.com/s/1whXOvudL8j6LfoGE19dEqg) (The key is 'ai3d').

## Results
The example results of MPU-Net algorithm is shown as follows:
![alt text](/show/result1.png "FESEM")
![alt text](/show/result2.png "OM")
