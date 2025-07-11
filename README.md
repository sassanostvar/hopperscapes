# HopperScapes
Mapping and generative modeling of forewing morphology, structure, and patterning in planthoppers.

## Data
Data curation for the HopperScapes project is ongoing. 

Part of the effort is focused on sampling established Northeast populations of Lycorma delicatula. 

At the same time, we are using online sources to assemble a pan-hemipteran dataset. 

### Local sources:
- Morningside Heights, New York City
- Hudson River Valley, New York

Locally sourced specimens are imaged using transmitted light microscopy. Specimen and image metadata are organized as defined in [hopper_vae.data.record.py](./hopper_vae/data/record.py).

### Web sources:

- Wikipedia Commons
- iNaturalist
- FLOW hemiptera databases

## Dataset structure
Light microscopy data are organized according to the [ome-zarr](https://github.com/ome/ome-zarr-py) specification:

    raw.zarr/
    └─ specimenID/                
        └─ forewing/
            ├─ left/                  
            │   ├─ rgb/   # (3×Y×X)
            │   └─ .attrs
            └─ right/
                ├─ rgb/
                └─ .attrs

## Segmentation
To study wing morphology, microstructure (venation), and pigmentation patterns, we segment the transmitted light microscopy images for wing outlines, veins, spots, and pigmentation domains. 


We use a multi-head network with a compact UNet-style encoder-decoder backbone and four logit heads for the respective semantic segmentation tasks [(hopper_vae/segmentation)](hopper_vae/segmentation).

## Roadmap
- [x] Implement data preprocessing pipeline for microscopy images
- [x] Implement segmentation model training and evaluation
- [x] Expand dataset with web-sourced images
- [ ] Implement data preprocessing pipeline for web-sourced images
- [ ] Implement generative model training and evaluation