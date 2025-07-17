# Segmentation

Quantitative analysis of wing morphology and patterning requires accurate localization of features of interest in imaging data. We pose the localization of structural features as a multi-task semantic segmentation problem. 

Depending on the species, some features, like the wing outline, spots, and pigmentation domains, are more or less straightforward to segment. 

Other features, like the venation network, can be challenging to segment when sample illumination is not optimal and the venation network is occluded by other features. 

Here, we use a multi-step, multi-resolution approach to the segmentation problem, as outlined below.

## Table of Contents
- [First-pass Multi-Task Semantic Segmentation](#first-pass-multi-task-semantic-segmentation)
    - [Network Architecture](#network-architecture)
    - [Loss Function](#loss-function)
    - [Training Dynamics](#training-dynamics)
    - [Performance](#performance)
    - [Usage](#usage)
        - [Datasets](#datasets)
        - [Data Augmentation](#data-augmentation)
        - [Training](#training)
        - [Inference](#inference)

## First-pass Multi-task Semantic Segmentation
### Network Architecture
The first segmentation model (`segmentation.models.HopperNetLite`) is a compact, multi-head UNet with approximately 100k trainable parameters and configurable heads supporting multiple semantic segmentation tasks. We train the model on downsampled and padded images (512x512 or 1024x1024), allowing the task-specific heads to share the same encoder and decoder. 

![Demo](../../assets/UNet_Lite.png)

A configurable multi-head architecture provides flexibility to experiment with feature sets across species.

![Demo](../../assets/sample_record.png)

We use a composite loss function to train the model on manual segmentations of (1) wing outlines/areas, (2) spots, (3) pigmentation domains, and (4) venation networks in _L. delicatula_ tegmina. 

We use the trained model in preprocessing to standardize the dataset for downstream analysis.

### Loss Function
The composite loss function is defined in [loss.py](../segmentation/loss.py) and configured in [configs.py](../configs.py) or via [configs.yaml](../configs.yaml). The loss function is configured to accommodate task-specific mask morphologies and class imbalance. We use binary cross-entropy (BCE) and soft Dice loss for wings, focal loss and soft Dice loss for spots, and a combination of BCE, soft Dice, and soft-cLDice for veins: 
- wing: bce + soft_dice
- spots: focal + soft_dice
- veins: bce + soft_dice + soft-clDice
- domains: ce (+ soft_dice)

We use the cross-entropy loss for pigmentation domains, which may vary in number in different species.

The loss function can be configured as follows:

```yaml
# example loss function configs
wing:
  bce:         {weight: 1.0, params: {pos_weight: 5.0}}
  soft_dice:   {weight: 1.0, params: {}}
veins:
  bce:         {weight: 0.3, params: {pos_weight: 50.0}}
  soft_dice:   {weight: 1.5, params: {}}
  cldice:      {weight: 1.5, params: {}}
spots:
  focal:       {weight: 1.0, params: {alpha: 0.85, gamma: 2.0}}
  soft_dice:   {weight: 1.0, params: {}}
domains:
  ce:          {weight: 0.7, params: {}}
  soft_dice:   {weight: 0.7, params: {}}
```
Full file: [configs.yaml](../configs.yaml).

### Training Dynamics
We use gradient clipping to promote stable early training. We also dynamically freeze and unfreeze the heads, based on preset Dice score thresholds, to ensure all task-specific heads receive sufficient gradients. 

### Performance
The wing Dice score saturates quickly, followed by the pigmentation domains and spots, in that order.

![Demo](../../assets/seg.gif)

The compact network trained on downsampled images struggles with the intricate venation network, which is occluded and lower in contrast than the other features. Addressing this limitation motivates the next model design iterations.

## Usage

### Datasets
Custom pytorch Dataset classes and collate functions are defined in [segmentation.dataset.py](../segmentation/dataset.py).

The model can be trained on data that are organized in a directory structure like the following: 

    └── root
        └── train
            ├── images
            │   └── <record_id>.jpg
            └── masks
                ├── domains
                │   └── <record_id>_seg_domains.tif
                ├── spots
                │   └── <record_id>_seg_spots.tif
                ├── veins
                │   └── <record_id>_seg_veins.tif
                └── wing
                    └── <record_id>_seg_wing.tif

The model can also be trained directly on ome-zarr stores that are prepared according to the specification in [data/zarr_store.py](../data/zarr_store.py). 

### Data Augmentation
Data augmentation utilities are included in [segmentation/augment/](../segmentation/augment/). Use the `augment` command-line interface to create augmented datasets locally:

```bash
% python -m pip install -e .
% python -m hopperscapes.segmentation.augment \
--images_dir $PATH_TO_IMAGES
--masks_dir $PATH_TO_MASKS
--savedir $PATH_TO_SAVE_AUGMENTED_SET
```

### Training
Train the model in a notebook or using the `train` command-line interface:

```bash
% python -m pip install -e .
% python -m hopperscapes.segmentation.train \ 
--configs_path $PATH_TO_CONFIGS_YAML \ 
--images_dir $PATH_TO_IMAGES \ 
--masks_dir $PATH_TO_MASKS \ 
--checkpoint_path $PATH_TO_PRETRAINED_CHECKPOINT
```
You can also train directly on ome-zarr stores.

### Inference
Apply a trained checkpoint to an image using the `infer` command-line interface:

```bash
% python -m pip install -e .
% python -m hopperscapes.segmentation.infer \ 
--image_path $PATH_TO_IMAGE \ 
--checkpoint_path $PATH_TO_CHECKPOINT \ 
--device "cuda"
```
