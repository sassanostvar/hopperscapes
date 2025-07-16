# Segmentation

Quantitative analysis of wing morphology and patterning requires accurate localization of the respective structures in imaging data. Localization of distinct structures can be posed as a multi-task semantic segmentation problem. 

Depending on the species, some features like the wing outline, spots, and pigmentation domains are more or less straightforward to segment. 

Other features like the venation network can be challenging to segment when sample illumination is not optimal and the venation network is occluded by other features. 

We use a multi-step multi-resolution approach to the segmentation problem as outlined below.

## Table of Contents
- [First-pass multi-task semantic segmentation](#first-pass-multi-task-semantic-segmentation)
    - [Loss function](#loss-function)
    - [Training dynamics](#training-dynamics)
    - [Performance](#performance)
    - [Usage](#usage)
        - [Datasets](#datasets)
        - [Data augmentation](#data-augmentation)
        - [Training](#training)
        - [Inference](#inference)

## First-pass multi-task semantic segmentation
The first segmentation model (segmentation.models.HopperNetLite) is a compact multi-head UNet with ~100k trainable parameters and configurable heads supporting multiple semantic segmentation tasks. We train the model on downsampled and padded images (512x512 or 1024x1024), allowing the heads to share the same encoder and decoder. We use this model in pre-processing to standardize the dataset for downstream analysis, including segmentation of finer structures using more advanced solutions.

We adopt supervised learning to train the model using a composite loss function for semantic segmentation of (1) wing outlines, (2) spots, (3) pigmentation domains, and (4) venation networks in _L. delicatula_ tegmina. 

![Demo](../../assets/sample_record.png)

The configurable multi-head architecture provides the flexibility to experiment with feature sets across taxa.

### Loss function
The composite loss function is defined in [loss.py](../segmentation/loss.py) and configured in [configs.py](../configs.py) or [configs.yaml](../configs.yaml). The criterion is designed to accommodate extreme class imbalance and task-specific morphological features. We use binary cross entropy (BCE) and soft Dice loss for wings, focal loss and soft dice loss for spots, and a combination of BCE, soft Dice, and soft-cLDice for veins. We use the cross entropy loss for pigmentation domains, which may vary in number in different species. 

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

### Training dynamics
We implement gradient clipping to promote stable training performance and dynamic freezing/unfreezing of heads based on preset threshold Dice scores to ensure all task-specific heads receive sufficient gradients. 

### Performance
In practice, the wing Dice score saturates quickly, followed by the pigmentation domains and spots, in that order. As expected, the compact network trained on downsampled images struggles with the intricate venation network, which is also occluded and lower in contrast.

![Demo](../../assets/seg.gif)

## Usage

### Datasets
We provide two interfaces to load training, validation, and testing data. The model can be trained directly on ome-zarr stores as specified in [data/zarr_store.py](../data/zarr_store.py). Alternatively, the model can be trained on images and masks that are organized in a directory structure like the following: 

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

The custom pytorch Dataset classes and collate functions are defined in [segmentation.dataset.py](../segmentation/dataset.py).

### Data augmentation
A growing suite of data augmentation utilities are included in [segmentation/augment/](../segmentation/augment/).

### Training
The model can be trained in a notebook or using the command-line interface:

```bash
(hopperscapes-env) % python -m hopperscapes.segmentation.train --configs_path $PATH_TO_CONFIGS_YAML --images_dir $PATH_TO_IMAGES --masks_dir $PATH_TO_MASKS --checkpoint_path $PATH_TO_PRETRAINED_CHECKPOINT
```
You can also train directly on ome-zarr stores.

### Inference
We can apply a trained checkpoint to a given image using the inference CLI as follows

```bash
(hopperscapes-env) % python -m hopperscapes.segmentation.infer --image_path $PATH_TO_IMAGE --checkpoint_path $PATH_TO_CHECKPOINT --device "cuda"

```
