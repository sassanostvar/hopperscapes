# Segmentation

Analysis of wing morphology and patterning poses a multiclass semantic segmentation problem. Some of the tasks like identifying the wing area, spots, and pigmentation domains are relatively trivial. Other tasks like segmenting the venation network are more challenging. 

We use a multi-step approach to the segmentation problem as outlined below.

## 1. First-pass multi-task semantic segmentation
The first segmentation model (models.HopperNetLite) is a compact UNet with ~100k trainable paramters for multi-task semantic segmentation. This model is trained on downsampled images and used during pre-processing to help standardize the dataset before fine-grained downstream analyses.

We adopt supervised learning using a composite loss function to train the model on four semantic segmentation tasks for wing outlines, spots, pigmentation domains, and venation. 

<p align="center">
<img src="../../assets/sample_record.png", style="max-width: 600px;">
</p>

The composite loss function is defined in [loss.py](loss.py) and configured in [configs.py](../configs.py).

This simple model manages to solve the wing, pigmentation domain, and spots segmentation problems, but struggles with the venation network.

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

## Roadmap

- [ ] Model architecture:
    - [ ] add trainable upsampling blocks
    - [ ] increase model depth 
    - [ ] test residual blocks 
    - [ ] test transformers for bottleneck 
- [ ] Training datasets:
    - [ ] implement multiprocessing for data agumentation pipeline
    - [ ] expand data augmentation with tile-wise color inversions 
    - [ ] expand data augmentation with focal gaussian blur
    - [ ] include other branching structures like leaves in training
