# MPU-Net
The Code of our paper MPU-Net

## Abstract
In material science, image segmentation is of great significance for the quantitative analysis of microstructures, which is a crucial step in building the processing-structure-properties relationship. Although deep learning has become the mainstream in the field of material microscopic image analysis, the complexity of the microstructure still creates challenges to the boundary detection task. For example, the precipitate always covers the grain boundary, resulting in the over-segmentation of grains and hindering the accurate characterization of microstructure.
In this work, we propose a novel method Multi-phase U-Net, dubbed as MPU-Net, to effectively detect the grain boundaries and precipitate at the same time in an end-to-end manner. In addition, we design two post-processing strategies to further improve boundary detection results. The adaptive grain redistribution strategy can rectify the over-segmentation errors of grain boundary caused by the cover of the precipitate phase. The pruning strategy can eliminate the non-closure boundary and improve the clarity of visualization results. Extensive experiment results on two classical datasets demonstrate that the proposed method achieves promising performance in both objective and subjective assessment. 

## Environment
> python 3.6
> pytorch 1.0
> gala (for evaluation)

gala is installed according to https://github.com/janelia-flyem/gala.

## Quick Start


Pre-train parameters download:
For wpunet segmentation, you can download at Baidu Pan (The key is '4yx7') or Google Drive, you should unzip it at './segmentation/'.
