import argparse
import os
import pickle
import random

import numpy as np
import matplotlib.pyplot as plt

from kirigami.structure import MatrixStructure
from kirigami.utils import plot_structure, find_invalid_quads, find_overlapping_quads
from offset_data_generator import (
    _compute_boundary_points_and_corners,
    _estimate_overlap_ratio,
    MAX_OVERLAP_RATIO,
)


def mask_extent_for_points(points, out_h, out_w):
    """Match the normalization used by the rasterizer for proper overlays."""
    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    sx = (out_w - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (out_h - 1) / (ymax - ymin) if ymax > ymin else 1.0
    s = min(sx, sy)
    return [xmin, xmin + out_w / s, ymax - out_h / s, ymax]


def reconstruct_shape_from_metadata(meta):
    """Rebuild structure geometry at phi=0 using stored interior offsets.

    Returns (structure, points_0) or (None, None) if reconstruction is not possible.
    """
    try:
        grid_rows = int(meta["grid_rows"])
        grid_cols = int(meta["grid_cols"])
        interior_offsets = np.asarray(meta["interior_offsets"], dtype=np.float32)
    except Exception:
        return None, None

    structure = MatrixStructure(num_linkage_rows=grid_rows, num_linkage_cols=grid_cols)
    boundary_points, corners = _compute_boundary_points_and_corners(structure)
    boundary_offsets = [[0.0] * grid_rows, [0.0] * grid_cols, [0.0] * grid_rows, [0.0] * grid_cols]

    structure.linear_inverse_design(
        np.vstack(boundary_points), corners, interior_offsets, boundary_offsets
    )
    structure.assign_node_layers()
    structure.assign_quad_genders()
    structure.make_hinge_contact_points()

    points_0, _ = structure.layout(0.0)
    # recentre to match generator/visualizer convention
    points_0[:, 0] -= (points_0[:, 0].max() + points_0[:, 0].min()) / 2.0
    points_0[:, 1] -= (points_0[:, 1].max() + points_0[:, 1].min()) / 2.0
    return structure, points_0


def pick_samples(ds_obj, split, n, seed):
    """Return a list of n samples from the dataset object.

    Supports either {"train": [...], "valid": [...]} or a flat list.
    """
    rng = random.Random(seed)
    if isinstance(ds_obj, dict) and split in ds_obj:
        pool = ds_obj[split]
    else:
        pool = ds_obj if isinstance(ds_obj, list) else []

    if not pool:
        raise RuntimeError("Dataset appears empty or unrecognized format.")

    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    idxs = idxs[: max(1, min(n, len(idxs)))]
    return [pool[i] for i in idxs]


def show_dataset_samples(
    dataset_path="kirigami_dataset.pkl",
    split="train",
    n_samples=6,
    seed=0,
    show_raw_mask=True,
    save_path=None,
):
    """Load dataset pickle and visualize a few sample (image, mask, overlay) triplets.

    - If geometry metadata is present, also reconstruct shape and overlay mask like visualizer.
    - Falls back to image + raw mask when reconstruction isn't possible.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, "rb") as f:
        ds = pickle.load(f)

    samples = pick_samples(ds, split, n_samples, seed)

    # Determine columns: image, raw mask, overlay (if possible)
    # We try to reconstruct at least for the first sample to decide layout.
    can_overlay = False
    first_meta = samples[0].get("metadata", {}) if isinstance(samples[0], dict) else {}
    if isinstance(first_meta, dict) and "interior_offsets" in first_meta:
        can_overlay = True

    cols = 3 if can_overlay and show_raw_mask else (2 if can_overlay else 2)
    fig, axes = plt.subplots(n_samples, cols, figsize=(3.6 * cols, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    for row, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise RuntimeError(
                "Unexpected sample format; expected dict with keys 'image' and 'mask'."
            )

        img = sample.get("image")
        msk = sample.get("mask")

        if img is None or msk is None:
            raise RuntimeError("Sample missing 'image' or 'mask' array.")

        # Squeeze to 2D for display
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[0] == 1:
            img2d = img[0]
        elif isinstance(img, np.ndarray) and img.ndim == 2:
            img2d = img
        else:
            # Unexpected; try to visualize interior_offsets from metadata if present
            img2d = np.asarray(sample.get("metadata", {}).get("interior_offsets", img))

        if isinstance(msk, np.ndarray) and msk.ndim == 3 and msk.shape[0] == 1:
            msk2d = msk[0]
        elif isinstance(msk, np.ndarray) and msk.ndim == 2:
            msk2d = msk
        else:
            # can't visualize
            raise RuntimeError("Mask has unexpected shape; expected (1,H,W) or (H,W).")

        # Try to reconstruct shape for overlay
        structure, pts = (None, None)
        if can_overlay:
            structure, pts = reconstruct_shape_from_metadata(sample.get("metadata", {}))
            if structure is None or pts is None:
                can_overlay = False  # fall back if reconstruction fails later

        # Decide axes
        if can_overlay and show_raw_mask:
            axI, axM, axO = axes[row, 0], axes[row, 1], axes[row, 2]
        elif can_overlay:
            axI, axO = axes[row, 0], axes[row, 1]
            axM = None
        else:
            axI, axM = axes[row, 0], axes[row, 1]
            axO = None

        # Image
        axI.imshow(img2d, cmap="gray", origin="upper")
        axI.set_title("Image")
        axI.axis("off")

        # Raw mask if requested/available
        if axM is not None:
            axM.imshow(msk2d, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            axM.set_title("Mask [0,1]")
            axM.axis("off")

        # Overlay like visualizer
        if can_overlay and axO is not None and structure is not None and pts is not None:
            img_h, img_w = msk2d.shape
            extent = mask_extent_for_points(pts, img_h, img_w)
            invalid_quads = find_invalid_quads(pts, structure.quads)
            overlaps = find_overlapping_quads(pts, structure.quads)
            ratio = _estimate_overlap_ratio(pts, structure.quads, msk2d, img_h, img_w)

            plot_structure(pts, structure.quads, linkages=None, ax=axO)
            label = "Shape + Mask"
            if invalid_quads:
                label += " (invalid quads)"
            if overlaps:
                label += " (overlaps)"
            if ratio > MAX_OVERLAP_RATIO:
                label += f" (overlap {ratio*100:.1f}%)"
            axO.set_title(label)

            # Darken outside the mask
            dark = np.zeros((img_h, img_w, 4), dtype=np.float32)
            dark[..., :3] = 0.0
            dark[..., 3] = (1.0 - msk2d) * 0.6
            axO.imshow(dark, extent=extent, origin="upper", zorder=5)

            # Green tint inside the mask
            overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
            overlay[..., 1] = 1.0
            overlay[..., 3] = msk2d * 0.35
            axO.imshow(overlay, extent=extent, origin="upper", zorder=6)
            axO.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        print(f"Saved preview to {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Quick dataset sampler and visualizer")
    parser.add_argument("--data", default="kirigami_dataset3.pkl", help="Path to dataset .pkl")
    parser.add_argument(
        "--split", default="train", choices=["train", "valid"], help="Split to sample"
    )
    parser.add_argument("--n", type=int, default=6, help="Number of samples to display")
    parser.add_argument("--seed", type=int, default=24, help="Random seed for sampling")
    parser.add_argument("--no-raw-mask", action="store_true", help="Hide raw mask column")
    parser.add_argument(
        "--save",
        default="checker_dataset.png",
        help="Optional path to save the figure instead of showing",
    )
    args = parser.parse_args()

    show_dataset_samples(
        dataset_path=args.data,
        split=args.split,
        n_samples=args.n,
        seed=args.seed,
        show_raw_mask=not args.no_raw_mask,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()
