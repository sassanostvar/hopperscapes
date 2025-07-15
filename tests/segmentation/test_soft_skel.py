from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import sknw
import torch
from skimage.io import imread
from skimage.morphology import binary_dilation, binary_erosion, disk, skeletonize, thin

from hopper_vae.segmentation.loss import soft_skel

TEST_IMAGE = (
    Path(__file__).parent.parent / "test_data" / "LD_F_TC_02024_0024_left_forewing.jpg"
)
TEST_VEINS_MASK = (
    Path(__file__).parent.parent
    / "test_data"
    / "LD_F_TC_02024_0024_left_forewing_seg_veins.tif"
)


@pytest.mark.unit
def test_soft_skel(debug=False):
    """
    Apply soft skeletonize to a sample veins seg mask.
    """
    img = imread(TEST_IMAGE)
    mask = imread(TEST_VEINS_MASK)
    #
    thinned_mask = thin(binary_erosion(mask))
    dilated_after_thin = binary_dilation(thinned_mask, disk(3))
    #
    skel = skeletonize(thinned_mask)
    sknw_graph = sknw.build_sknw(skel, multi=False)
    #
    mask_t = torch.from_numpy(mask).unsqueeze(0).float()
    veins_soft_skel_t = soft_skel(mask_t)
    veins_soft_skel = veins_soft_skel_t.detach().cpu().numpy().squeeze(0)
    #
    print(f"tp: {(mask*veins_soft_skel).sum()}, total: {mask.sum()}")

    if debug:
        fig, ax = plt.subplots(3, 2, figsize=(10, 10))
        ax[0][0].imshow(img)
        ax[0][0].set_title("Original Image")
        ax[0][1].imshow(mask)
        ax[0][1].set_title("Veins Mask")
        ax[1][0].imshow(veins_soft_skel, cmap="gray")
        ax[1][0].set_title("Soft Skeleton")
        ax[1][1].imshow(mask - veins_soft_skel, cmap="gray")
        ax[1][1].set_title("Veins Mask - Soft Skeleton")
        #
        ax[2][0].imshow(thinned_mask, cmap="gray")
        ax[2][0].set_title("Thinned Mask")
        ax[2][1].imshow(dilated_after_thin, cmap="gray")
        ax[2][1].set_title("Dilated After Thin")
        # overlay the graph on dilated_after_thin

        # draw edges by pts
        for s, e in sknw_graph.edges():
            try:
                ps = sknw_graph[s][e]["pts"]
                ax[2][1].plot(ps[:, 1], ps[:, 0], "green")
            except KeyError:
                print(f"KeyError: {s} {e}")

        # draw node by o
        nodes = sknw_graph.nodes()
        ps = np.array([nodes[i]["o"] for i in nodes])
        ax[2][1].plot(ps[:, 1], ps[:, 0], "r.")

        for a in ax.flat:
            a.axis("off")
        plt.tight_layout()
        plt.show()
