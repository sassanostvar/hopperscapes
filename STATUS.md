# HopperScapes: Development Status and Roadmap

Current version: 0.1.0<br>
Release status: Alpha

HopperScapes is in active development.

## Alpha (0.1.0)
Completed:
- [x] Implement sample collection and imaging metadata schemas
- [x] Implement Zarr stores for microscopy data
- [x] Implement first semantic segmentation model for microscopy data
- [x] Collect first set of web-sourced images

In Progress:
- [ ] Prepare light microscopy dataset for 1st raw imaging data release.
- [ ] Implement image pre- and post-processing pipelines for microscopy data
- [ ] Implement image pre- and post-processing pipelines for web-sourced data
- [ ] Implement basic morphometry pipelines
- [ ] Expand test suite to ~85% converage

## Roadmap
- Experiments:
    - [ ] Continue sample collection.
    - [ ] Complete imaging experiments for the 2nd raw imaging data release.

- Web-sourced datasets:
    - [ ] Implement data pre-processing pipeline for web-sourced images.

- Segmentation:
    - [ ] Prepare training dataset for wing venation segmentation.
    - [ ] Expand segmentation model to improve performance on the venation segmentation task.

- Representation learning:
    - [ ] Implement self-supervised representation learning for segmented light microscopy data.

## Contribuing/Feedback
To contribute to the project, feel free to:
- Open an issue
- Start a discussion in GitHub Discussions
- Reach out to the maintainer via GitHub