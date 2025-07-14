# Segmentation

## Initial semantic segmentation
The first segmentation model is a compact UNet with ~100k trainable paramters for multi-task semantic segmentation. This model is trained on downsampled images and used during pre-processing to help standardize the dataset for more fine-grained downstream analyses.

We adopt supervised learning using a composite loss function to train the model on four semantic segmentation tasks for wing outlines, spots, pigmentation domains, and venation. The composite loss function is defined in [loss.py](loss.py).


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
