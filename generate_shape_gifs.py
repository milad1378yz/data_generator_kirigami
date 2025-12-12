import argparse
import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt

from optimize_eps_shapes import _build_structure_context
from kirigami.utils import plot_structure, rotate_points


def _center_points(points: np.ndarray, phi: float) -> np.ndarray:
    pts = rotate_points(points, np.array([0, 0]), -(np.pi - phi) / 2.0)
    pts[:, 0] -= (pts[:, 0].max() + pts[:, 0].min()) / 2.0
    pts[:, 1] -= (pts[:, 1].max() + pts[:, 1].min()) / 2.0
    return pts


def _prep_structure(eps_field: np.ndarray, grid_rows: int, grid_cols: int):
    ctx = _build_structure_context(grid_cols, grid_rows)
    structure = ctx["structure"]
    boundary_points_vector = ctx["boundary_points_vector"]
    corners = ctx["corners"]
    boundary_offsets = ctx["boundary_offsets"]

    structure.linear_inverse_design(
        boundary_points_vector, corners, eps_field, boundary_offsets
    )
    structure.assign_node_layers()
    structure.assign_quad_genders()
    structure.make_hinge_contact_points()
    return structure


def make_gif(name, eps_field, grid_rows, grid_cols, out_dir, n_frames=40, duration=0.12):
    os.makedirs(out_dir, exist_ok=True)
    structure = _prep_structure(eps_field, grid_rows, grid_cols)

    phis = np.linspace(np.pi, 0.0, n_frames)
    all_pts = []
    for phi in phis:
        pts, _ = structure.layout(phi)
        pts = _center_points(pts, phi)
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)
    pad = 0.05 * max(np.ptp(all_pts[:, 0]), np.ptp(all_pts[:, 1]))
    xlim = (all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ylim = (all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)

    frames = []
    for idx, phi in enumerate(phis):
        pts, _ = structure.layout(phi)
        pts = _center_points(pts, phi)

        fig, ax = plt.subplots(figsize=(5, 5))
        plot_structure(pts, structure.quads, structure.linkages, ax=ax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        fig.tight_layout(pad=0.05)

        tmp_path = os.path.join(out_dir, f"{name}_frame_{idx:03d}.png")
        fig.savefig(tmp_path, dpi=150)
        plt.close(fig)
        frames.append(tmp_path)

    gif_path = os.path.join(out_dir, f"{name}_animation.gif")
    images = [imageio.imread(f) for f in frames]
    imageio.mimsave(gif_path, images, duration=duration, loop=0)

    for f in frames:
        os.remove(f)

    print(f"Saved {gif_path}")
    return gif_path


def main():
    parser = argparse.ArgumentParser(description="Generate square-to-shape GIFs from optimized eps.")
    parser.add_argument("--npz", default="optimized_eps_shapes.npz", help="Path to optimized shapes npz")
    parser.add_argument("--out", default="optimized_gifs", help="Output directory for GIFs")
    parser.add_argument("--frames", type=int, default=40, help="Number of frames per GIF")
    parser.add_argument("--duration", type=float, default=0.12, help="Seconds per frame in GIF")
    args = parser.parse_args()

    npz = np.load(args.npz)
    grid_rows = int(npz["grid_rows"])
    grid_cols = int(npz["grid_cols"])

    shape_names = [k.replace("_eps", "") for k in npz.files if k.endswith("_eps")]
    for name in shape_names:
        eps = npz[f"{name}_eps"]
        make_gif(name, eps, grid_rows, grid_cols, args.out, n_frames=args.frames, duration=args.duration)


if __name__ == "__main__":
    main()
