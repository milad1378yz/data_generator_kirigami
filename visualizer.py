import numpy as np
import matplotlib.pyplot as plt

from Structure import MatrixStructure
from Utils import plot_structure, find_invalid_quads
from offset_data_generator import (
    _compute_boundary_points_and_corners,
    _rasterize_quads_filled,
    _map_u_to_eps,
    EPS_MIN,
    EPS_MAX,
    EPS_SCALE,
)


def generate_shape_and_mask(grid_rows=18, grid_cols=18, img_h=128, img_w=128, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Build structure and boundary constraints
    structure = MatrixStructure(num_linkage_rows=grid_rows, num_linkage_cols=grid_cols)
    boundary_points, corners = _compute_boundary_points_and_corners(structure)

    # Interior offsets using same mapping as the generator
    u = rng.random((grid_rows, grid_cols))
    interior_offsets = _map_u_to_eps(u, EPS_MIN, EPS_MAX, EPS_SCALE)

    # Zero boundary offsets, as in the generator
    boundary_offsets = [[0.0] * grid_rows, [0.0] * grid_cols, [0.0] * grid_rows, [0.0] * grid_cols]

    # Inverse design + housekeeping
    structure.linear_inverse_design(
        np.vstack(boundary_points), corners, interior_offsets, boundary_offsets
    )
    structure.assign_node_layers()
    structure.assign_quad_genders()
    structure.make_hinge_contact_points()

    # Layout at φ=0 for silhouette and recentre (matches generator)
    points_0, _ = structure.layout(0.0)
    points_0[:, 0] -= (points_0[:, 0].max() + points_0[:, 0].min()) / 2.0
    points_0[:, 1] -= (points_0[:, 1].max() + points_0[:, 1].min()) / 2.0

    # Silhouette mask using the same rasterizer
    mask = _rasterize_quads_filled(points_0, structure.quads, out_h=img_h, out_w=img_w)

    return structure, points_0, mask


def mask_extent_for_points(points, out_h, out_w):
    # Match the normalization used inside _rasterize_quads_filled
    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    sx = (out_w - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (out_h - 1) / (ymax - ymin) if ymax > ymin else 1.0
    s = min(sx, sy)
    # Pixel edges map to world edges: [xmin, xmin + out_w/s] x [ymax - out_h/s, ymax]
    return [xmin, xmin + out_w / s, ymax - out_h / s, ymax]


def show_samples(
    n_samples=3,
    grid_rows=16,
    grid_cols=16,
    img_h=128,
    img_w=128,
    alpha=0.35,
    dark_alpha=0.6,
    show_raw_mask=True,
    seed=0,
):
    rng = np.random.default_rng(seed)
    cols = 3 if show_raw_mask else 2
    fig, axes = plt.subplots(n_samples, cols, figsize=(3.5 * cols, 3 * n_samples))
    if n_samples == 1:
        axes = np.array([axes])

    for i in range(n_samples):
        structure, pts, mask = generate_shape_and_mask(grid_rows, grid_cols, img_h, img_w, rng)
        extent = mask_extent_for_points(pts, img_h, img_w)
        invalid_quads = find_invalid_quads(pts, structure.quads)

        if show_raw_mask:
            axM, ax0, ax1 = axes[i, 0], axes[i, 1], axes[i, 2]
        else:
            ax0, ax1 = axes[i, 0], axes[i, 1]

        # Optional: raw mask visualization to verify 0/1 values
        if show_raw_mask:
            axM.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0, origin="upper")
            axM.set_title("Raw Mask [0,1]")
            axM.axis("off")

        # Middle/Left: shape only
        plot_structure(pts, structure.quads, linkages=None, ax=ax0)
        ax0.set_title("Shape" + (" (invalid)" if invalid_quads else ""))

        # Right: shape + transparent mask overlay with darkening outside mask
        plot_structure(pts, structure.quads, linkages=None, ax=ax1)
        ax1.set_title("Shape + Mask Overlay" + (" (invalid)" if invalid_quads else ""))

        # Darken non-mask area so non-mask shows as 0 (dark)
        dark = np.zeros((img_h, img_w, 4), dtype=np.float32)
        dark[..., :3] = 0.0  # black
        dark[..., 3] = (1.0 - mask) * dark_alpha
        ax1.imshow(dark, extent=extent, origin="upper", zorder=5)

        # Overlay mask as transparent tint (visible where mask==1)
        overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
        overlay[..., 1] = 1.0  # green tint
        overlay[..., 3] = mask * alpha  # alpha from mask
        ax1.imshow(overlay, extent=extent, origin="upper", zorder=6)
        if invalid_quads:
            issues = ", ".join(sorted({reason for _, reason in invalid_quads}))
            print(f"[warn] Sample {i}: {len(invalid_quads)} invalid quads detected -> {issues}")

    plt.tight_layout()
    plt.savefig("shape_and_mask_samples.png", dpi=300)


if __name__ == "__main__":
    # Example usage (generates a gallery for manual inspection)
    show_samples(n_samples=10, grid_rows=10, grid_cols=8, img_h=256, img_w=256, alpha=0.4, seed=12)
