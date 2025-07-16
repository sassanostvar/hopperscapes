# HOPPERSCAPES


![Python](https://img.shields.io/badge/python-3.9+-blue)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Tests](https://img.shields.io/badge/tests-pytest-green)
![Coverage](https://img.shields.io/badge/coverage-71%25-yellowgreen)
![Status](https://img.shields.io/badge/status-alpha-orange)

> ** Demo Alpha Release - July 2025 **
## Overview

<p align='center'>Mapping and representation learning of forewing morphology and patterning in planthoppers</p>

<p align="center">
<img src="assets/wings.gif", style="max-width: 300px;">
</p>

Planthoppers have evolved remarkably intricate and diverse forewing compositions. [HopperScapes](https://github.com/sassanostvar/hopperscapes/tree/main) is an ongoing effort toward representation learning of the underlying _morpho-chromosapce_.

This repository provides the components of an end-to-end pipeline for quantitative analysis of light microscopy and photographic images of tegmina, including utilities for dataset management, image processing, semantic segmentation, and morphometry.

## Repository Structure
[HopperScapes](https://github.com/sassanostvar/hopperscapes/tree/main) is a growing toolset for dataset management, image pre-processing, semantic segmentation, post-processing, and quantification of planthopper forewing compositions. The repository is organized as follows:

    .
    ├── hopperscapes
    │   ├── data
    │   ├── imageproc
    │   ├── segmentation
    │   ├── morphometry
    │   └── vae # WIP
    │
    ├── assets
    ├── docs
    ├── notebooks
    ├── scripts
    └── tests

The components, functions, and implementation details of each module can be found in the respective READMEs.

## Usage

A typical pipeline for light microscopy data is as follows:

<p align="center"> <code>Preprocessing -> Segmentation → Postprocessing → Morphometry</code> </p>


## Segmentation
Quantitative analysis of wing morphology, microstructure (venation), and pigmentation patterns relies on accurate segmentation of the respective features in images. See [hopper_vae/segmentation/README.md](hopper_vae/segmentation/README.md) for details of segmentation model design and training.

## Data
Data curation for the HopperScapes project is ongoing. Part of the effort is focused on sampling established Northeast populations of _Lycorma delicatula_. Locally sourced specimens are imaged using transmitted light microscopy. Specimen collection and imaging metadata are organized as specified in [hopper_vae.data.record.py](./hopper_vae/data/record.py).

Initial release of the first light microscopy dataset is planned for Summer 2025.

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


## Data Sources
Local sources include Morningside Heights, New York City and Hudson River Valley, New York.

Web sources include Wikipedia Commons, iNaturalist, and FLOW hemiptera databases.


## Current Status & Roadmap

HopperScapes is in active development. The planned next steps for Summer 2025 are: 

- Proof-reading and releasing the first light microscopy dataset of _Lycorma delicatula_ specimens along with multi-feature segmentations and model checkpoints;
- Benchmarking self-supervised pre-training for the segmentation model;
- Improving model performance on the ventation network segmentation task.

Please see [STATUS.md](STATUS.md) for more details. 

## Credits
[HopperScapes](https://github.com/sassanostvar/hopperscapes/tree/main) was conceived, designed, planned, and executed by [Sassan Ostvar](https://sassanostvar.github.io).

## Contact
Contributions and collaborations are most welcome. Please reach out to [Sassan Ostvar](https://sassanostvar.github.io).

## Citation
_forthcoming_

## Related Work
- Ronellenfitsch, Henrik, Jana Lasser, Douglas C. Daly, and Eleni Katifori. "Topological phenotypes constitute a new dimension in the phenotypic space of leaf venation networks." PLoS computational biology 11, no. 12 (2015): e1004680.
- Hoffmann, Jordan, Seth Donoughe, Kathy Li, Mary K. Salcedo, and Chris H. Rycroft. "A simple developmental model recapitulates complex insect wing venation patterns." Proceedings of the National Academy of Sciences 115, no. 40 (2018): 9905-9910.
- Katifori, Eleni. "The transport network of a leaf." Comptes Rendus. Physique 19, no. 4 (2018): 244-252.
- Salcedo, Mary K., Jordan Hoffmann, Seth Donoughe, and L. Mahadevan. "Computational analysis of size, shape and structure of insect wings." Biology Open 8, no. 10 (2019): bio040774.
- Lürig, Moritz D., Seth Donoughe, Erik I. Svensson, Arthur Porto, and Masahito Tsuboi. "Computer vision, machine learning, and the promise of phenomics in ecology and evolutionary biology." Frontiers in Ecology and Evolution 9 (2021): 642774.
