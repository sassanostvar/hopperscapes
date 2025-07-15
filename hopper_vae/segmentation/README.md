# Segmentation

Quantitative analysis of wing morphology and patterning requires accurate localization of the respective structures in imaging data. Identification of distinct structures can be posed as a multi-task semantic segmentation problem. 

Depending on the species, some of the features like the wing outline, spots, and pigmentation domains are more or less trivial to segment. 

Other features like the venation network can be challenging to segment when sample illumination is not optimal and the venation network is occluded by the other features. 

We use a multi-step multi-resolution approach to the segmentation problem as outlined below.

## 1. First-pass multi-task semantic segmentation
The first segmentation model (models.HopperNetLite) is a compact multi-head UNet with ~100k trainable paramters and configurable heads supporting multiple semantic segmentation tasks. We train the model on downsampled and padded images (512x512 or 1024x1024) and use the model in pre-processing to standardize the dataset for downstream analysis.

We adopt supervised learning to train the model using a composite loss function for semantic segmentation tasks targeting (1) wing outlines, (2) spots, (3) pigmentation domains, and (4) venation networks. 

<p align="center">
<img src="../../assets/sample_record.png", style="max-width: 600px;">
</p>

### Loss function
A composite loss function is defined in [loss.py](loss.py) and configured in [configs.py](../configs.py). The criterion is designed to accommodate extreme class imbalance and task-specific morphological features. We use binary cross entropy (BCE) and soft Dice loss for wings, focal loss and soft dice loss for spots, BCE, soft Dice, and soft-cLDice for veins, and cross entropy loss for pigmentation domains. 

### Training
To ensure stable training performance, we implement gradient clipping and dynamic freezing/unfreezing of heads based on preset threshold Dice scores during model training. 

### Performance
In practice, the wing Dice scores saturates quickly, followed by the pigmentation domains and spots, in that order. As expected, HopperNetLite struggles with the venation network.

## Datasets
The train, valid, and test sets are organized as follows: 

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

The associated pytorch Dataset and custom collate functions are defined in [dataset.py](dataset.py).
