import time
import os
import numpy as np
import scipy.optimize as scopt
from numpy.linalg import norm
import matplotlib.pyplot as plt

from Structure import MatrixStructure
from offset_data_generator import _rasterize_quads_filled
import imageio.v2 as imageio
from Utils import plot_structure


def _build_structure_context(width: int, height: int):
    structure = MatrixStructure(num_linkage_rows=height, num_linkage_cols=width)
    bound_linkage_inds = [structure.get_boundary_linkages(i) for i in range(4)]
    bound_directions = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])
    boundary_points = []
    corners = []
    for i, bound in enumerate(bound_linkage_inds):
        local_boundary_points = []
        for j, linkage_ind in enumerate(bound):
            p = structure.is_linkage_parallel_to_boundary(linkage_ind[0], linkage_ind[1], i)
            if j == 0:
                corner = np.array([linkage_ind[1], -linkage_ind[0]]) + bound_directions[i]
                if not p:
                    corner += bound_directions[(i - 1) % 4]
                corners.append(corner)
            if not p:
                point = np.array([linkage_ind[1], -linkage_ind[0]]) + bound_directions[i]
                local_boundary_points.append(point)
        boundary_points.append(np.vstack(local_boundary_points))
    corners = np.vstack(corners)
    boundary_offsets = [[0.0] * height, [0.0] * width, [0.0] * height, [0.0] * width]
    boundary_points_vector = np.vstack(boundary_points)

    # reference contracted state to pick usable boundary nodes
    structure.linear_inverse_design(
        boundary_points_vector,
        corners,
        np.reshape(np.zeros(width * height), (height, width)),
        boundary_offsets,
    )
    structure.make_hinge_contact_points()
    deployed_points, _ = structure.layout(phi=0.0)
    dual_bound_inds = []
    for bound_ind in range(4):
        dual_bound_inds.extend(structure.get_dual_boundary_node_inds(bound_ind))
    reduced_dual_bound_inds = []
    for i, ind in enumerate(dual_bound_inds):
        next_i = (i + 1) % len(dual_bound_inds)
        next_ind = dual_bound_inds[next_i]
        if norm(deployed_points[ind] - deployed_points[next_ind]) > 1e-10:
            reduced_dual_bound_inds.append(ind)

    return {
        "structure": structure,
        "boundary_points_vector": boundary_points_vector,
        "corners": corners,
        "boundary_offsets": boundary_offsets,
        "reduced_dual_bound_inds": reduced_dual_bound_inds,
        "width": width,
        "height": height,
    }


def _center_points(points: np.ndarray) -> np.ndarray:
    pts = points.copy()
    pts[:, 0] -= (pts[:, 0].max() + pts[:, 0].min()) / 2
    pts[:, 1] -= (pts[:, 1].max() + pts[:, 1].min()) / 2
    return pts


def _regular_polygon_radius(theta: np.ndarray, n_sides: int, radius: float = 1.0) -> np.ndarray:
    sector = 2.0 * np.pi / float(n_sides)
    local = np.mod(theta, sector)
    denom = np.cos(local - sector / 2.0)
    denom = np.clip(denom, 1e-8, None)
    return (radius * np.cos(np.pi / float(n_sides))) / denom


def _contour_to_radial_fn(contour_pts: np.ndarray):
    """
    Build a radial function r(theta) from a closed contour defined by samples.
    Uses the support function: r(theta) = max_p <p, dir(theta)>.
    """
    pts = np.asarray(contour_pts, dtype=np.float64)
    pts = pts - pts.mean(axis=0, keepdims=True)
    radii = np.linalg.norm(pts, axis=1)
    scale = np.mean(radii)
    pts = pts / max(scale, 1e-6)

    def r(theta: np.ndarray) -> np.ndarray:
        dirs = np.column_stack([np.cos(theta), np.sin(theta)])
        dots = dirs @ pts.T
        out = dots.max(axis=1)
        return np.clip(out, 1e-4, None)

    return r


def _make_heart_contour(n_samples: int = 720) -> np.ndarray:
    """
    Parametric heart curve sampled and normalized.
    Equation source: classic heart-shaped curve.
    """
    t = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = (
        13 * np.cos(t)
        - 5 * np.cos(2 * t)
        - 2 * np.cos(3 * t)
        - np.cos(4 * t)
    )
    pts = np.column_stack([x, y])
    pts = pts - pts.mean(axis=0, keepdims=True)
    pts = pts / np.mean(np.linalg.norm(pts, axis=1))
    return pts


def _make_boundary_residual(context, target_radius_fn):
    width = context["width"]
    height = context["height"]
    structure = context["structure"]
    boundary_points_vector = context["boundary_points_vector"]
    corners = context["corners"]
    boundary_offsets = context["boundary_offsets"]
    reduced_dual_bound_inds = context["reduced_dual_bound_inds"]

    def residual(interior_offsets_vector):
        interior_offsets = np.reshape(interior_offsets_vector, (height, width))
        structure.linear_inverse_design(
            boundary_points_vector, corners, interior_offsets, boundary_offsets
        )
        structure.make_hinge_contact_points()
        deployed_points, _ = structure.layout(phi=0.0)
        boundary_pts = deployed_points[reduced_dual_bound_inds]

        center = np.mean(boundary_pts, axis=0)
        rel = boundary_pts - center
        angles = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
        r_actual = np.linalg.norm(rel, axis=1)

        target_r = target_radius_fn(angles)
        scale = np.mean(r_actual) / np.mean(target_r)
        return r_actual - scale * target_r

    return residual


def _compute_layout_and_mask(context, interior_offsets: np.ndarray, mask_size: int):
    structure = context["structure"]
    boundary_points_vector = context["boundary_points_vector"]
    corners = context["corners"]
    boundary_offsets = context["boundary_offsets"]

    structure.linear_inverse_design(boundary_points_vector, corners, interior_offsets, boundary_offsets)
    structure.assign_node_layers()
    structure.assign_quad_genders()
    structure.make_hinge_contact_points()

    points_0, _ = structure.layout(phi=0.0)
    points_0 = _center_points(points_0)
    mask = _rasterize_quads_filled(points_0, structure.quads, out_h=mask_size, out_w=mask_size)
    return points_0, mask, structure


def _mask_extent_for_points(points, out_h, out_w):
    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    sx = (out_w - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (out_h - 1) / (ymax - ymin) if ymax > ymin else 1.0
    s = min(sx, sy)
    return [xmin, xmin + out_w / s, ymax - out_h / s, ymax]


def _save_overlay(name, points, mask, structure, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    img_h, img_w = mask.shape
    extent = _mask_extent_for_points(points, img_h, img_w)

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_structure(points, structure.quads, linkages=None, ax=ax)

    dark = np.zeros((img_h, img_w, 4), dtype=np.float32)
    dark[..., :3] = 0.0
    dark[..., 3] = (1.0 - mask) * 0.6
    ax.imshow(dark, extent=extent, origin="upper", zorder=5)

    overlay = np.zeros((img_h, img_w, 4), dtype=np.float32)
    overlay[..., 1] = 1.0
    overlay[..., 3] = mask * 0.35
    ax.imshow(overlay, extent=extent, origin="upper", zorder=6)

    ax.set_title(f"{name} – deformed mask overlay")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"{name}_overlay.png"), dpi=200)
    plt.close(fig)


def optimize_shape(
    target_name: str,
    target_radius_fn,
    width=10,
    height=10,
    mask_size=128,
    max_nfev=120,
    verbose=1,
):
    context = _build_structure_context(width, height)
    residual_fn = _make_boundary_residual(context, target_radius_fn)

    lower = np.full(width * height, -0.8)
    upper = np.full(width * height, 1.0)

    print(f"Optimizing for {target_name}...")
    start = time.time()
    result = scopt.least_squares(
        residual_fn,
        np.zeros(width * height),
        bounds=(lower, upper),
        xtol=1e-8,
        ftol=1e-8,
        gtol=1e-8,
        max_nfev=max_nfev,
        verbose=verbose,
    )
    duration = time.time() - start
    eps_matrix = np.reshape(result.x, (height, width)).astype(np.float32)
    points, mask, _ = _compute_layout_and_mask(context, eps_matrix, mask_size)
    return {
        "eps": eps_matrix,
        "mask": mask.astype(np.float32),
        "cost": float(result.cost),
        "time_sec": duration,
        "status": result.status,
        "message": result.message,
        "points": points,
        "structure": context["structure"],
    }


def main():
    width = 10
    height = 10
    mask_size = 128
    max_nfev = 120
    verbose = 1

    heart_radial_fn = _contour_to_radial_fn(_make_heart_contour())

    shapes = {
        "circle": lambda theta: np.ones_like(theta, dtype=np.float64),
        "hexagon": lambda theta: _regular_polygon_radius(theta, 6),
        "heart": heart_radial_fn,
        "star": lambda theta: np.maximum(0.6, 1.0 + 0.25 * np.cos(5 * theta)),
        "clover": lambda theta: np.maximum(0.4, 1.0 + 0.25 * np.cos(3 * theta)),
    }

    results = {}
    for name, fn in shapes.items():
        results[name] = optimize_shape(
            name,
            fn,
            width,
            height,
            mask_size,
            max_nfev=max_nfev,
            verbose=verbose,
        )

    save_kwargs = {
        "grid_rows": height,
        "grid_cols": width,
        "mask_size": mask_size,
    }
    for name, res in results.items():
        save_kwargs[f"{name}_eps"] = res["eps"]
        save_kwargs[f"{name}_mask"] = res["mask"]
        save_kwargs[f"{name}_cost"] = res["cost"]

    np.savez_compressed("optimized_eps_shapes.npz", **save_kwargs)

    # Save per-shape masks (visual and raw)
    os.makedirs("optimized_masks", exist_ok=True)
    for name, res in results.items():
        mask = res["mask"]
        np.save(os.path.join("optimized_masks", f"{name}_mask.npy"), mask)
        imageio.imwrite(
            os.path.join("optimized_masks", f"{name}_mask.png"), (mask * 255).astype(np.uint8)
        )
        _save_overlay(name, res["points"], mask, res["structure"], "optimized_masks")

    print("Saved optimized eps and masks ➜ optimized_eps_shapes.npz")
    for name, res in results.items():
        print(f"{name:8s} cost {res['cost']:.6f} in {res['time_sec']:.2f}s (status {res['status']})")


if __name__ == "__main__":
    main()
