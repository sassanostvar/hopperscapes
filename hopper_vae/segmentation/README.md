# Segmentation

The first segmentation model is a compact UNet with ~100k trainable paramters. This model is trained on downsampled images and used during pre-processing to help standardize the dataset before more fine-grained analyses.

We use a composite loss function to train the model on four semantic segmentation tasks for wing outlines, spots, pigmentation domains, and venation. The loss function is defined in [loss.py](loss.py).


## Roadmap

- [ ] add trainable upsampling blocks
- [ ] increase depth 
- [ ] test residual blocks 
- [ ] test transformers for bottleneck 
- [ ] expand data augmentation with block-wise color inversions 
- [ ] include other branching structures in trainings sets 
