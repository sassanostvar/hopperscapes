# HOPPERSCAPES

![Tests](https://img.shields.io/badge/tests-pytest-green)
![Coverage](https://img.shields.io/badge/coverage-31%25-red)

## Overview

Mapping and representation learning of forewing morphology and patterning in planthoppers.

<p align="center">
<img src="assets/wings.gif", style="max-width: 300px;">
</p>

Planthoppers show remarkably diverse forewing structure and patterning. [HopperScapes](https://github.com/sassanostvar/hopperscapes/tree/main) is an ongoing effort toward representation learning of the underlying _morpho-chromosapce_.

The current repository includes components of an end-to-end pipeline for record management, image pre-processing, semantic segmentation, and morphometry.

## Data
Data curation for the HopperScapes project is ongoing. Part of the effort is focused on sampling established Northeast populations of _Lycorma delicatula_. Locally sourced specimens are imaged using transmitted light microscopy. Specimen collection and imaging metadata are organized as specified in [hopper_vae.data.record.py](./hopper_vae/data/record.py).

## Dataset structure
Light microscopy data are organized using [ome-zarr](https://github.com/ome/ome-zarr-py) and the following specification defined in [hopper_vae/data/zarr.py](hopper_vae/data/zarr.py):

    raw.zarr/
    └─ specimenID/                
        └─ forewing/
            ├─ left/                  
            │   ├─ rgb/   # (3×Y×X)
            │   └─ .attrs
            └─ right/
                ├─ rgb/
                └─ .attrs

Initial release of the first light microscopy dataset is planned for Summer 2025.

## Repository Structure
    .
    ├── hopperscapes
    │   ├── data
    │   ├── imageproc
    │   ├── segmentation
    │   ├── morphometry
    │   └── vae
    │
    ├── assets
    ├── docs
    ├── notebooks
    ├── scripts
    └── tests

## Segmentation
To study wing morphology, microstructure (venation), and pigmentation patterns, we segment the transmitted light microscopy images for wing outlines, veins, spots, and pigmentation domains. 

For model implementation and training details, see [hopper_vae/segmentation/README.md](hopper_vae/segmentation/README.md).

## Data Sources
### Local sources:
- Morningside Heights, New York City
- Hudson River Valley, New York

### Web sources:

- Wikipedia Commons
- iNaturalist
- FLOW hemiptera databases


## Roadmap

HopperScapes is in active development. Please see [STATUS.md](STATUS.md) for details.

## Credits
[HopperScapes](https://github.com/sassanostvar/hopperscapes/tree/main) was conceived, planned, and executed by [Sassan Ostvar](sassanostvar.github.io).

## Contact
Contributions and collaborations are most welcome. Please reach out to [Sassan Ostvar](sassanostvar.github.io).

## Related Work
- Ronellenfitsch, Henrik, Jana Lasser, Douglas C. Daly, and Eleni Katifori. "Topological phenotypes constitute a new dimension in the phenotypic space of leaf venation networks." PLoS computational biology 11, no. 12 (2015): e1004680.
- Hoffmann, Jordan, Seth Donoughe, Kathy Li, Mary K. Salcedo, and Chris H. Rycroft. "A simple developmental model recapitulates complex insect wing venation patterns." Proceedings of the National Academy of Sciences 115, no. 40 (2018): 9905-9910.
- Katifori, Eleni. "The transport network of a leaf." Comptes Rendus. Physique 19, no. 4 (2018): 244-252.
- Salcedo, Mary K., Jordan Hoffmann, Seth Donoughe, and L. Mahadevan. "Computational analysis of size, shape and structure of insect wings." Biology Open 8, no. 10 (2019): bio040774.
- Lürig, Moritz D., Seth Donoughe, Erik I. Svensson, Arthur Porto, and Masahito Tsuboi. "Computer vision, machine learning, and the promise of phenomics in ecology and evolutionary biology." Frontiers in Ecology and Evolution 9 (2021): 642774.
