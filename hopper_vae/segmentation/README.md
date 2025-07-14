# Segmentation

## Initial semantic segmentation
The first segmentation model (models.HopperNetLite) is a compact UNet with ~100k trainable paramters for multi-task semantic segmentation. This model is trained on downsampled images and used during pre-processing to help standardize the dataset before fine-grained downstream analyses.

We adopt supervised learning using a composite loss function to train the model on four semantic segmentation tasks for wing outlines, spots, pigmentation domains, and venation. 

<p align="center">
<img src="../../assets/sample_record.png", style="max-width: 600px;">
</p>

The composite loss function is defined in [loss.py](loss.py) and configured in [configs.py](../configs.py).


## Roadmap

- [ ] Model architecture:
    - [ ] add trainable upsampling blocks
    - [ ] increase model depth 
    - [ ] test residual blocks 
    - [ ] test transformers for bottleneck 
- [ ] Training datasets:
    - [ ] expand data augmentation with tile-wise color inversions 
    - [ ] expand data augmentation with focal gaussian blur
    - [ ] include other branching structures like leaves in training
