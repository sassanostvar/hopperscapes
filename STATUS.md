# HopperScapes: Development Status and Roadmap

Current version: 0.1.0<br>
Release status: Alpha

HopperScapes is in active development.

## Alpha (0.1.0)
Completed:
- [x] Implement sample collection and imaging metadata schemas [(hopperscapes/data/record.py)](hopperscapes/data/record.py)
- [x] Implement Zarr stores for microscopy data [(hopperscapes/data/zarr_store.py)](hopperscapes/data/zarr_store.py)
- [x] Implement semantic segmentation models for automated pre-processing of light microscopy data [(hopperscapes/segmentation)](hopperscapes/segmentation)
- [x] Implement image preprocessing and standardization pipelines for light microscopy data
- [x] Curate the first set of web-sourced images
- [x] Implement basic morphometry pipelines [(hopperscapes/morphometry)](hopperscapes/morphometry)

In Progress:
- [ ] Expand test suite to ~85% coverage

## Project Roadmap
- Experiments & Datasets:
    - [ ] Continue sample collection in Q3 2025.
    - [ ] Complete imaging experiments for the 2nd batch of local samples.
    - [ ] Prepare the first segmented and proof-read batch of light microscopy dataset for public release.
    - [ ] Implement image pre- and post-processing pipelines for web-sourced data

- Segmentation:
    - [ ] Prepare train, valid, and test datasets for wing venation segmentation.
    - [ ] Expand segmentation model to improve performance on the venation segmentation task.
        - [ ] Benchmark deeper segmentation models.
        - [ ] Benchmark pre-trained encoders.
    
- Representation learning:
    - [ ] Implement self-supervised representation learning for segmented light microscopy data.

## Contributing/Feedback

To contribute to the project, feel free to:
- Open an issue
- Reach out to the maintainer via GitHub