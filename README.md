# HOPPERSCAPES
Mapping and representation learning of forewing morphology and patterning in planthoppers.

<p align="center">
<img src="assets/wings.gif", style="max-width: 300px;">
</p>

## Data
Data curation for the HopperScapes project is ongoing. Part of the effort is focused on sampling established Northeast populations of _Lycorma delicatula_. Locally sourced specimens are imaged using transmitted light microscopy. Specimen collection and imaging metadata are organized as specified in [hopper_vae.data.record.py](./hopper_vae/data/record.py).

## Dataset structure
Light microscopy data are organized using [ome-zarr](https://github.com/ome/ome-zarr-py) and the following specification:

    raw.zarr/
    └─ specimenID/                
        └─ forewing/
            ├─ left/                  
            │   ├─ rgb/   # (3×Y×X)
            │   └─ .attrs
            └─ right/
                ├─ rgb/
                └─ .attrs

## Data Sources
### Local sources:
- Morningside Heights, New York City
- Hudson River Valley, New York

### Web sources:

- Wikipedia Commons
- iNaturalist
- FLOW hemiptera databases

## Segmentation
To study wing morphology, microstructure (venation), and pigmentation patterns, we segment the transmitted light microscopy images for wing outlines, veins, spots, and pigmentation domains. 

For model implementation and training details, see [hopper_vae/segmentation/README.md](hopper_vae/segmentation/README.md).

## Roadmap

HopperScapes is in active development. Please see [STATUS.md](STATUS.md) for details.

## Contact
[Sassan Ostvar](sassanostvar.github.io)