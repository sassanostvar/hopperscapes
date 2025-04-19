# spotted-lanternflies
Generative modeling of wing morphology, structure, and patterning in spotted lanternflies and other planthoppers.

## segmentation
We are interested in extracting from transmission light microscopy images wing outlines, venation structures, spots, and overall pigmentation. We use a multi-head segmentation network with a compact UNet-style encoder-decoder backbone and four logit heads for outline, veins, spots, and ... ?

## repository structure
    wing-vae/
    ├── data/
    │   ├── raw/                   # Original JPEGs, masks, metadata
    │   ├── processed/             # Cleaned masks, aligned outlines, numpy arrays
    │   ├── samples/               # Debug samples or small training subsets
    │   └── metadata/              # JSON, CSV, or pickle of parsed Metadata objects
    │
    ├── notebooks/
    │   ├── 01_explore_data.ipynb          # Visualize raw inputs 
    │   ├── 02_clean_align_masks.ipynb     # From microSAM to binary outlines
    │   ├── 03_pilot_train_vae.ipynb       # Early training loop tests
    │   └── 04_latent_viz_analysis.ipynb   # UMAP, cluster visualizations
    │
    ├── src/
    │   ├── data/
    │   │   ├── loader.py          # `WingPatternDataset` class
    │   │   └── preprocess.py      # Alignment, binarization, metadata parsing
    │   ├── model/
    │   │   ├── conditional_vae.py # VAE architecture (encoders, decoder)
    │   │   └── loss.py            # Custom β-VAE loss functions
    │   ├── train/
    │   │   └── train_vae.py       # Standalone training script
    │   ├── utils/
    │   │   ├── viz.py             # Helper functions for overlay, plotting
    │   │   └── metrics.py         # Dice, BCE, KL divergence, etc.
    │   └── config.py              # Hyperparameters and paths
    │
    ├── outputs/
    │   ├── logs/                  # Training logs or tensorboard files
    │   ├── reconstructions/       # Sample reconstructions
    │   └── latent_space/          # UMAPs, latent plots
    │
    ├── README.md
    ├── requirements.txt
    ├── .gitignore
    └── LICENSE (optional)

## references