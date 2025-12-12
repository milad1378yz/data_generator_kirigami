import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from matplotlib.path import Path
import matplotlib.patches as mpatches

from Structure import MatrixStructure
from Utils import plot_structure, rotate_points


# the pattern size
width = 14
height = 14

# create positive Gaussian-like random matrices with mean ~1 (resample negatives)
mean = 1.0
std = 0.3
np.random.seed(0)


def positive_gaussian(shape, mean=1.0, std=0.3):
    m = np.random.normal(loc=mean, scale=std, size=shape)
    mask = m <= 0
    while np.any(mask):
        m[mask] = np.random.normal(loc=mean, scale=std, size=np.count_nonzero(mask))
        mask = m <= 0
    # optionally force exact mean 1 across the matrix
    m *= mean / np.mean(m)
    return m


def _mask_extent_and_scale(points, out_h, out_w):
    x = points[:, 0]
    y = points[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    sx = (out_w - 1) / (xmax - xmin) if xmax > xmin else 1.0
    sy = (out_h - 1) / (ymax - ymin) if ymax > ymin else 1.0
    s = min(sx, sy)
    extent = [xmin, xmin + out_w / s, ymax - out_h / s, ymax]
    return extent, s, xmin, xmax, ymin, ymax


def rasterize_quads(points, quads, out_h=512, out_w=512):
    extent, s, xmin, xmax, ymin, ymax = _mask_extent_and_scale(points, out_h, out_w)
    verts = []
    codes = []
    for quad in quads:
        poly = points[np.asarray(quad)]
        for xy in poly:
            xp = (xy[0] - xmin) * s
            yp = (ymax - xy[1]) * s
            verts.append((xp, yp))
        verts.append((0.0, 0.0))  # CLOSEPOLY ignores this vertex
        codes.extend([Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY])

    compound_path = Path(verts, codes)
    xs = np.arange(out_w) + 0.5
    ys = np.arange(out_h) + 0.5
    xv, yv = np.meshgrid(xs, ys)
    grid_pts = np.column_stack([xv.ravel(), yv.ravel()])
    mask = compound_path.contains_points(grid_pts).reshape(out_h, out_w).astype(np.float32)
    return mask, extent, s, xmin, xmax, ymin, ymax


def _pixel_center_to_world(rc, xmin, xmax, ymin, ymax, s):
    r, c = rc
    xp = c + 0.5
    yp = r + 0.5
    xw = xp / s + xmin
    yw = ymax - yp / s
    return np.array([xw, yw])


def _pixel_bbox_to_world(bbox, xmin, xmax, ymin, ymax, s):
    r0, r1, c0, c1 = bbox
    corners = [
        _pixel_center_to_world((r0, c0), xmin, xmax, ymin, ymax, s),
        _pixel_center_to_world((r0, c1), xmin, xmax, ymin, ymax, s),
        _pixel_center_to_world((r1, c0), xmin, xmax, ymin, ymax, s),
        _pixel_center_to_world((r1, c1), xmin, xmax, ymin, ymax, s),
    ]
    xs = [p[0] for p in corners]
    ys = [p[1] for p in corners]
    return min(xs), max(xs), min(ys), max(ys)


def negative_space_mask(filled_mask):
    solid = filled_mask > 0.5
    h, w = solid.shape
    visited = np.zeros_like(solid, dtype=bool)
    outside = np.zeros_like(solid, dtype=bool)
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if (not solid[r, c]) and (not visited[r, c]):
                visited[r, c] = True
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if (not solid[r, c]) and (not visited[r, c]):
                visited[r, c] = True
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (not solid[nr, nc]) and (not visited[nr, nc]):
                visited[nr, nc] = True
                outside[nr, nc] = True
                q.append((nr, nc))

    holes = (~solid) & (~outside)
    return holes


def connected_components(binary_mask):
    h, w = binary_mask.shape
    visited = np.zeros_like(binary_mask, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if binary_mask[r, c] and (not visited[r, c]):
                q = deque([(r, c)])
                visited[r, c] = True
                pixels = []
                while q:
                    rr, cc = q.popleft()
                    pixels.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and binary_mask[nr, nc]
                            and (not visited[nr, nc])
                        ):
                            visited[nr, nc] = True
                            q.append((nr, nc))
                pixels = np.array(pixels)
                bbox = (
                    pixels[:, 0].min(),
                    pixels[:, 0].max(),
                    pixels[:, 1].min(),
                    pixels[:, 1].max(),
                )
                components.append({"pixels": pixels, "center": pixels.mean(axis=0), "bbox": bbox})
    return components


a_matrix = positive_gaussian((height, width), mean, std)
b_matrix = positive_gaussian((height, width), mean, std)

interior_offsets = a_matrix / b_matrix - 1.0


# create a square kirigami structure
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


# get the resulting random kirigami pattern using the linear inverse design method
structure.linear_inverse_design(
    np.vstack(boundary_points), corners, interior_offsets, boundary_offsets
)
structure.assign_node_layers()
structure.assign_quad_genders()
structure.make_hinge_contact_points()

phi = np.pi / 2  # mid deployment
deployed_points, _ = structure.layout(phi)
deployed_points = rotate_points(deployed_points, np.array([0, 0]), -(np.pi - phi) / 2.0)

deployed_points[:, 0] = (
    deployed_points[:, 0] - (np.max(deployed_points[:, 0]) + np.min(deployed_points[:, 0])) / 2
)
deployed_points[:, 1] = (
    deployed_points[:, 1] - (np.max(deployed_points[:, 1]) + np.min(deployed_points[:, 1])) / 2
)


mask_res = 600
filled_mask, _, scale, xmin, xmax, ymin, ymax = rasterize_quads(
    deployed_points, structure.quads, out_h=mask_res, out_w=mask_res
)
holes = negative_space_mask(filled_mask)
components = connected_components(holes)
main_hole = max(components, key=lambda c: len(c["pixels"])) if components else None

if main_hole is not None:
    hole_center_world = _pixel_center_to_world(main_hole["center"], xmin, xmax, ymin, ymax, scale)
    hole_box_world = _pixel_bbox_to_world(main_hole["bbox"], xmin, xmax, ymin, ymax, scale)
else:
    hole_center_world = np.zeros(2)
    hole_box_world = (-1, 1, -1, 1)

zoom_w = hole_box_world[1] - hole_box_world[0]
zoom_h = hole_box_world[3] - hole_box_world[2]
annot_pad = max(zoom_w, zoom_h) * 0.2
annot_xlim = (hole_center_world[0] - annot_pad, hole_center_world[0] + annot_pad)
annot_ylim = (hole_center_world[1] - annot_pad, hole_center_world[1] + annot_pad)

quad_centroids = np.array([deployed_points[q].mean(axis=0) for q in structure.quads])
quad_diffs = quad_centroids - hole_center_world
quadrant_masks = [
    (quad_diffs[:, 0] <= 0) & (quad_diffs[:, 1] <= 0),  # a: bottom-left
    (quad_diffs[:, 0] <= 0) & (quad_diffs[:, 1] >= 0),  # b: top-left
    (quad_diffs[:, 0] >= 0) & (quad_diffs[:, 1] >= 0),  # c: top-right
    (quad_diffs[:, 0] >= 0) & (quad_diffs[:, 1] <= 0),  # d: bottom-right
]

selected_quads = []
for mask in quadrant_masks:
    cand_inds = np.where(mask)[0]
    if len(cand_inds) > 0:
        local_dists = np.linalg.norm(quad_diffs[cand_inds], axis=1)
        selected_quads.append(cand_inds[np.argmin(local_dists)])
    else:
        remaining = [i for i in range(len(quad_centroids)) if i not in selected_quads]
        local_dists = np.linalg.norm(quad_diffs[remaining], axis=1)
        selected_quads.append(remaining[int(np.argmin(local_dists))])

selected_nodes = []
for qi in selected_quads:
    quad_nodes = structure.quads[qi]
    pts = deployed_points[quad_nodes]
    dists = np.linalg.norm(pts - hole_center_world, axis=1)
    for idx in np.argsort(dists):
        node_idx = quad_nodes[idx]
        if node_idx not in selected_nodes:
            selected_nodes.append(node_idx)
            break

if len(selected_nodes) < 4:
    remaining = [i for i in range(len(deployed_points)) if i not in selected_nodes]
    rem_pts = deployed_points[remaining]
    dists = np.linalg.norm(rem_pts - hole_center_world, axis=1)
    for idx in np.argsort(dists):
        selected_nodes.append(remaining[idx])
        if len(selected_nodes) == 4:
            break

node_pts = deployed_points[selected_nodes]
angles = np.arctan2(node_pts[:, 1] - hole_center_world[1], node_pts[:, 0] - hole_center_world[0])
ordering = np.argsort(angles)
node_pts = node_pts[ordering]

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Overview with highlighted negative spaces
plot_structure(deployed_points, structure.quads, structure.linkages, axs[0])
zoom_rect = mpatches.Rectangle(
    (annot_xlim[0], annot_ylim[0]),
    annot_xlim[1] - annot_xlim[0],
    annot_ylim[1] - annot_ylim[0],
    linewidth=2,
    edgecolor="tab:red",
    facecolor="none",
    zorder=6,
)
axs[0].add_patch(zoom_rect)
# axs[0].set_title("Full pattern at mid deployment", color="k")
axs[0].axis("off")
axs[0].set_aspect("equal")

# Annotated negative space with a,b,c,d and hinge points
plot_structure(deployed_points, structure.quads, structure.linkages, axs[1])
axs[1].set_xlim(*annot_xlim)
axs[1].set_ylim(*annot_ylim)

# Connect rectangle to zoomed view
for corner in [(annot_xlim[1], annot_ylim[1]), (annot_xlim[1], annot_ylim[0])]:
    target = (annot_xlim[0], corner[1])
    con = mpatches.ConnectionPatch(
        xyA=corner,
        xyB=target,
        coordsA="data",
        coordsB="data",
        axesA=axs[0],
        axesB=axs[1],
        linestyle="--",
        linewidth=1.4,
        color="tab:red",
        alpha=0.9,
        zorder=7,
    )
    con.set_clip_on(False)
    fig.add_artist(con)

for idx, pt in enumerate(node_pts):
    axs[1].scatter(pt[0], pt[1], color="dodgerblue", s=35, zorder=9)
    direction = pt - hole_center_world
    dist = np.linalg.norm(direction)
    if dist < 1e-9:
        direction = np.array([1.0, 0.0])
        dist = 1.0
    offset_len = 0.2 * dist + 0.03
    offset_vec = direction / dist * offset_len

    callibrator = np.array([-0.05, 0.05])
    offset_vec += callibrator

    text_pos = pt + offset_vec
    ha = "left" if direction[0] >= 0 else "right"
    va = "bottom" if direction[1] >= 0 else "top"
    axs[1].text(
        text_pos[0],
        text_pos[1],
        rf"$x_{{ij}}^{{{idx}}}$",
        color="navy",
        fontsize=16,
        weight="bold",
        ha=ha,
        va=va,
    )

for label, qi in zip(["a", "b", "c", "d"], selected_quads):
    cx, cy = quad_centroids[qi]
    centroid = np.array([cx, cy])
    to_center = hole_center_world - centroid
    dist = np.linalg.norm(to_center)
    if dist > 1e-9:
        to_center = to_center / dist
    label_pos = centroid + to_center * (0.25 * dist)
    axs[1].text(
        label_pos[0],
        label_pos[1],
        rf"${label}_{{ij}}$",
        color="darkred",
        fontsize=16,
        weight="bold",
        ha="center",
        va="center",
    )

phi_origin = node_pts[0]
vec_prev = node_pts[-1] - phi_origin
vec_next = node_pts[1] - phi_origin
theta_start = np.rad2deg(np.arctan2(vec_prev[1], vec_prev[0])) % 360
theta_target = np.rad2deg(np.arctan2(vec_next[1], vec_next[0])) % 360
arc_span = (theta_target - theta_start) % 360
if arc_span > 180:
    theta_start = theta_target
    arc_span = 360 - arc_span
theta_end = theta_start + arc_span
phi_radius = 0.55 * min(np.linalg.norm(vec_prev), np.linalg.norm(vec_next))
axs[1].add_patch(
    mpatches.Arc(
        xy=phi_origin,
        width=2 * phi_radius,
        height=2 * phi_radius,
        angle=0,
        theta1=theta_start,
        theta2=theta_end,
        color="k",
        linewidth=2.0,
        zorder=9,
    )
)
mid_angle = np.deg2rad(theta_start + arc_span / 2.0)
phi_label_pos = phi_origin + 0.65 * phi_radius * np.array([np.cos(mid_angle), np.sin(mid_angle)])
axs[1].text(
    phi_label_pos[0],
    phi_label_pos[1],
    r"$\phi_{ij}$",
    color="k",
    fontsize=16,
    weight="bold",
    zorder=10,
)

# axs[1].set_title("Annotated negative space (a,b,c,d + nodes)", color="k")
axs[1].axis("off")
axs[1].set_aspect("equal")

plt.tight_layout()
plt.savefig("negative_space_view.pdf", dpi=300)
