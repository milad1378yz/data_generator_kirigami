import argparse
import os
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np

from kirigami.utils import read_obj


ConnectorDef = Tuple[Tuple[int, int, int], Tuple[int, int, int], float, float]


def _normalize_to_origin(points: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Translate points so the lower-left of the bounding box is at (0, 0).

    Returns:
        points0: translated copy of input points
        width:   bbox width in model units
        height:  bbox height in model units
    """
    pts = np.asarray(points, dtype=float).copy()
    xmin = float(pts[:, 0].min())
    ymin = float(pts[:, 1].min())
    pts[:, 0] -= xmin
    pts[:, 1] -= ymin
    width = float(pts[:, 0].max())
    height = float(pts[:, 1].max())
    return pts, width, height


def _edge_key(
    p0: np.ndarray,
    p1: np.ndarray,
    scale: float,
    quant: float = 1e-5,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Quantized, orientation-independent key for a geometric edge.
    """
    q = max(quant * scale, 1e-9)

    def _qpt(p: np.ndarray) -> Tuple[int, int]:
        return (int(round(float(p[0]) / q)), int(round(float(p[1]) / q)))

    a = _qpt(p0)
    b = _qpt(p1)
    return (a, b) if a <= b else (b, a)


def _gather_segments(
    pts: np.ndarray, quads: np.ndarray
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Collect oriented edge segments for every quad.
    """
    segments: List[Tuple[np.ndarray, np.ndarray]] = []
    for quad in quads:
        ids = [int(i) for i in quad]
        loop = ids + ids[:1]
        for a, b in zip(loop[:-1], loop[1:]):
            segments.append((pts[a], pts[b]))
    return segments


def _unique_segments_for_cuts(
    pts: np.ndarray, quads: np.ndarray, scale: float
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Deduplicate geometric edges (directionless) for drawing the cut lines.
    """
    key2seg: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[np.ndarray, np.ndarray]] = {}
    segments = _gather_segments(pts, quads)
    for p0, p1 in segments:
        key = _edge_key(p0, p1, scale)
        if key not in key2seg:
            key2seg[key] = (p0, p1)
    return list(key2seg.values())


def _segment_intersection_with_params(
    a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray, tol: float = 1e-9
) -> Optional[Tuple[np.ndarray, float, float]]:
    """
    Return (point, t_a, t_b) for the intersection of segments a(t)=a0+t*(a1-a0), b(u)=b0+u*(b1-b0).

    Handles true crossings and collinear overlaps by picking the midpoint of the overlap.
    """
    va = a1 - a0
    vb = b1 - b0
    rxs = va[0] * vb[1] - va[1] * vb[0]
    q_p = b0 - a0
    qpxva = q_p[0] * va[1] - q_p[1] * va[0]

    if abs(rxs) < tol:
        if abs(qpxva) > tol:
            return None  # Parallel, non-overlapping
        # Collinear: project B onto A and take midpoint of the overlap range.
        vv = float(np.dot(va, va))
        if vv <= tol:
            return None
        t0 = float(np.dot(b0 - a0, va) / vv)
        t1 = float(np.dot(b1 - a0, va) / vv)
        t_min, t_max = sorted([t0, t1])
        if t_max < -tol or t_min > 1.0 + tol:
            return None
        t_clamp_low = max(0.0, t_min)
        t_clamp_high = min(1.0, t_max)
        t_mid = 0.5 * (t_clamp_low + t_clamp_high)
        p = a0 + t_mid * va
        vb_norm = float(np.dot(vb, vb))
        u_mid = float(np.dot(p - b0, vb) / max(vb_norm, tol))
        return p, t_mid, u_mid

    t = (q_p[0] * vb[1] - q_p[1] * vb[0]) / rxs
    u = qpxva / rxs
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        p = a0 + t * va
        return p, t, u
    return None


def _build_connector_defs_from_layout(
    pts: np.ndarray, quads: np.ndarray, tol: float = 1e-8
) -> List[ConnectorDef]:
    """
    Build connector definitions using a reference layout.

    Each connector is stored as:
        (edge_key_a, edge_key_b, t_a, t_b)
    where edge_key = (quad_index, min(v0, v1), max(v0, v1)) and t_a is the
    parametric position along edge A (0..1).
    """
    pts = np.asarray(pts, dtype=float)
    quads = np.asarray(quads, dtype=int)

    edges: List[Tuple[int, int, int, int, int]] = []  # (quad_idx, v0, v1, g0, g1)
    for q_idx, quad in enumerate(quads):
        ids = [int(i) for i in quad]
        loop = ids + ids[:1]
        for a, b in zip(loop[:-1], loop[1:]):
            edges.append((q_idx, a, b, min(a, b), max(a, b)))

    defs: Dict[Tuple[Tuple[int, int, int], Tuple[int, int, int]], Tuple[float, float]] = {}

    for i in range(len(edges)):
        qa, a0, a1, ga0, ga1 = edges[i]
        pa0 = pts[a0]
        pa1 = pts[a1]
        for j in range(i + 1, len(edges)):
            qb, b0, b1, gb0, gb1 = edges[j]
            if qa == qb:
                continue
            pb0 = pts[b0]
            pb1 = pts[b1]
            res = _segment_intersection_with_params(pa0, pa1, pb0, pb1, tol=tol)
            if res is None:
                continue
            _, t_a, t_b = res
            edge_key_a = (qa, ga0, ga1)
            edge_key_b = (qb, gb0, gb1)
            pair_key = tuple(sorted([edge_key_a, edge_key_b]))
            if pair_key not in defs:
                defs[pair_key] = (t_a, t_b)

    connector_defs: List[ConnectorDef] = []
    for (edge_key_a, edge_key_b), (t_a, t_b) in defs.items():
        connector_defs.append((edge_key_a, edge_key_b, float(t_a), float(t_b)))
    return connector_defs


def _evaluate_connector_points(
    connector_defs: List[ConnectorDef],
    pts: np.ndarray,
    quads: np.ndarray,
    tol: float = 1e-9,
    quant: float = 1e-5,
    dedup: bool = True,
) -> List[np.ndarray]:
    """
    Given connector definitions and a layout (pts), evaluate connector locations.

    Uses stored parametric positions along each edge so the connectors persist
    even if edges become collinear or nearly overlapping at extreme phis.
    """
    pts = np.asarray(pts, dtype=float)
    quads = np.asarray(quads, dtype=int)

    # Build lookup from edge key to endpoints
    edge_lookup: Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]] = {}
    for q_idx, quad in enumerate(quads):
        ids = [int(i) for i in quad]
        loop = ids + ids[:1]
        for a, b in zip(loop[:-1], loop[1:]):
            key = (q_idx, min(a, b), max(a, b))
            if key not in edge_lookup:
                edge_lookup[key] = (pts[a], pts[b])

    raw_points: List[np.ndarray] = []

    pts_bbox = np.asarray(pts, dtype=float)
    scale = max(
        float(pts_bbox[:, 0].max() - pts_bbox[:, 0].min()),
        float(pts_bbox[:, 1].max() - pts_bbox[:, 1].min()),
        1.0,
    )
    q = max(quant * scale, 1e-12)

    def _key(p: np.ndarray) -> Tuple[int, int]:
        return (int(round(float(p[0]) / q)), int(round(float(p[1]) / q)))

    for edge_a, edge_b, t_a, t_b in connector_defs:
        pa = edge_lookup.get(edge_a)
        pb = edge_lookup.get(edge_b)
        if pa is None or pb is None:
            continue
        a0, a1 = pa
        b0, b1 = pb
        va = a1 - a0
        vb = b1 - b0
        a_len = float(np.dot(va, va))
        b_len = float(np.dot(vb, vb))
        p_a = a0 + float(t_a) * va
        p_b = b0 + float(t_b) * vb
        if a_len < tol and b_len < tol:
            continue
        if a_len < tol:
            raw_points.append(p_b)
        elif b_len < tol:
            raw_points.append(p_a)
        else:
            raw_points.append(0.5 * (p_a + p_b))

    if not dedup:
        return raw_points

    connectors: Dict[Tuple[int, int], np.ndarray] = {}
    for p in raw_points:
        connectors[_key(p)] = p
    return list(connectors.values())


def _phi_to_radians(phi_value: float, phi_in_degrees: bool = False) -> float:
    """
    Interpret phi either as radians (default) or degrees.

    If phi is large (greater than about 2*pi) we auto-assume degrees to be
    forgiving for callers that pass 90 instead of np.pi/2.
    """
    phi_value = float(phi_value)
    if phi_in_degrees or abs(phi_value) > 2.0 * np.pi * 1.01:
        return np.deg2rad(phi_value)
    return phi_value


def _write_basic_r12_dxf(
    cut_lines: Iterable[Tuple[float, float, float, float]],
    connector_circles: Iterable[Tuple[float, float, float]],
    out_path: str,
) -> None:
    """
    Minimal R12 DXF writer with separate layers:
      - CUT:        line entities for full-depth cuts (borders + negative spaces)
      - CONNECTORS: circle entities for tiny reference dots (do NOT cut)
    Coordinates are assumed to already be in millimeters.
    """
    dxf: List[str] = []
    dxf.extend(
        [
            "0",
            "SECTION",
            "2",
            "HEADER",
            "9",
            "$ACADVER",
            "1",
            "AC1009",
            "999",
            "Units: millimeters; connectors on CONNECTORS layer are reference dots (skip cutting).",
            "0",
            "ENDSEC",
            "0",
            "SECTION",
            "2",
            "TABLES",
            "0",
            "TABLE",
            "2",
            "LAYER",
            "70",
            "2",  # number of layers
            "0",
            "LAYER",
            "2",
            "CUT",
            "70",
            "0",
            "62",
            "1",
            "6",
            "CONTINUOUS",
            "0",
            "LAYER",
            "2",
            "CONNECTORS",
            "70",
            "0",
            "62",
            "3",
            "6",
            "CONTINUOUS",
            "0",
            "ENDTAB",
            "0",
            "ENDSEC",
            "0",
            "SECTION",
            "2",
            "ENTITIES",
        ]
    )

    for x1, y1, x2, y2 in cut_lines:
        dxf.extend(
            [
                "0",
                "LINE",
                "8",
                "CUT",
                "10",
                f"{x1:.6f}",
                "20",
                f"{y1:.6f}",
                "30",
                "0",
                "11",
                f"{x2:.6f}",
                "21",
                f"{y2:.6f}",
                "31",
                "0",
            ]
        )

    for cx, cy, r in connector_circles:
        dxf.extend(
            [
                "0",
                "CIRCLE",
                "8",
                "CONNECTORS",
                "10",
                f"{cx:.6f}",
                "20",
                f"{cy:.6f}",
                "30",
                "0",
                "40",
                f"{r:.6f}",
            ]
        )

    dxf.extend(["0", "ENDSEC", "0", "EOF"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(dxf))


def _build_structure_context_for_eps(
    width: int, height: int
) -> Tuple["MatrixStructure", np.ndarray, np.ndarray, List[List[float]]]:
    from kirigami.structure import MatrixStructure

    structure = MatrixStructure(num_linkage_rows=int(height), num_linkage_cols=int(width))
    bound_linkage_inds = [structure.get_boundary_linkages(i) for i in range(4)]
    bound_directions = np.array([[-1.0, 0.0], [0.0, -1.0], [1.0, 0.0], [0.0, 1.0]])

    boundary_points: List[np.ndarray] = []
    corners: List[np.ndarray] = []
    for i, bound in enumerate(bound_linkage_inds):
        local_boundary_points: List[np.ndarray] = []
        for j, linkage_ind in enumerate(bound):
            is_parallel = structure.is_linkage_parallel_to_boundary(
                linkage_ind[0], linkage_ind[1], i
            )
            if j == 0:
                corner = np.array([linkage_ind[1], -linkage_ind[0]], dtype=float) + bound_directions[i]
                if not is_parallel:
                    corner += bound_directions[(i - 1) % 4]
                corners.append(corner)
            if not is_parallel:
                point = np.array([linkage_ind[1], -linkage_ind[0]], dtype=float) + bound_directions[i]
                local_boundary_points.append(point)
        boundary_points.append(
            np.vstack(local_boundary_points) if local_boundary_points else np.zeros((0, 2))
        )

    corners_arr = np.vstack(corners) if corners else np.zeros((0, 2))
    boundary_points_vector = (
        np.vstack(boundary_points) if boundary_points else np.zeros((0, 2))
    )
    boundary_offsets = [[0.0] * height, [0.0] * width, [0.0] * height, [0.0] * width]
    return structure, boundary_points_vector, corners_arr, boundary_offsets


def _build_structure_from_eps(eps: np.ndarray) -> "MatrixStructure":
    eps = np.asarray(eps, dtype=float)
    if eps.ndim != 2:
        raise ValueError(f"Expected eps as a 2D array, got shape {eps.shape}.")
    height, width = eps.shape
    structure, boundary_points_vector, corners, boundary_offsets = _build_structure_context_for_eps(
        width, height
    )
    structure.linear_inverse_design(boundary_points_vector, corners, eps, boundary_offsets)
    structure.assign_node_layers()
    structure.assign_quad_genders()
    structure.make_hinge_contact_points()
    return structure


class EpsLaserExporter:
    """
    Cache-friendly exporter for a single eps field across many deployment angles (phis).
    """

    def __init__(
        self,
        eps: np.ndarray,
        *,
        phi_ref: Optional[float] = None,
        phi_ref_in_degrees: bool = False,
    ) -> None:
        self.eps = np.asarray(eps, dtype=float)
        self.structure = _build_structure_from_eps(self.eps)

        if phi_ref is None:
            phi_ref = np.pi / 2.0
            phi_ref_in_degrees = False

        phi_ref_rad = _phi_to_radians(phi_ref, phi_in_degrees=phi_ref_in_degrees)
        pts_ref, _ = self.structure.layout(phi_ref_rad)
        pts_ref = np.asarray(pts_ref, dtype=float)[:, :2]
        self.connector_defs = _build_connector_defs_from_layout(pts_ref, self.structure.quads)

    def _layout_flat(
        self, phi_flat: float, *, phi_in_degrees: bool = False
    ) -> Tuple[np.ndarray, float, float, List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
        phi_radians = _phi_to_radians(phi_flat, phi_in_degrees=phi_in_degrees)
        points_flat, _ = self.structure.layout(phi_radians)
        pts2d = np.asarray(points_flat, dtype=float)[:, :2]

        pts0, width_bb, height_bb = _normalize_to_origin(pts2d)
        if width_bb <= 0.0 or height_bb <= 0.0:
            raise ValueError("Degenerate pattern: zero-area bounding box for eps pattern.")

        shape_scale = max(width_bb, height_bb, 1.0)
        cut_segments = _unique_segments_for_cuts(pts0, self.structure.quads, scale=shape_scale)

        is_extreme_phi = (abs(phi_radians) < 1e-6) or (abs(abs(phi_radians) - np.pi) < 1e-6)
        connector_points = _evaluate_connector_points(
            self.connector_defs, pts0, self.structure.quads, dedup=not is_extreme_phi
        )
        return pts0, width_bb, height_bb, cut_segments, connector_points

    def export_svg(
        self,
        svg_path: str,
        *,
        phi_flat: float = np.pi,
        cut_width: float = 0.02,
        hinge_width: float = 0.01,
        phi_in_degrees: bool = False,
        connector_radius: Optional[float] = None,
        preview_svg_path: Optional[str] = None,
    ) -> None:
        pts0, width_bb, height_bb, cut_segments, connector_points = self._layout_flat(
            phi_flat, phi_in_degrees=phi_in_degrees
        )

        def _to_svg_xy(p: np.ndarray) -> Tuple[float, float]:
            x = float(p[0])
            y = float(height_bb - p[1])
            return x, y

        cut_lines: List[Tuple[float, float, float, float]] = []
        for p0, p1 in cut_segments:
            x1, y1 = _to_svg_xy(p0)
            x2, y2 = _to_svg_xy(p1)
            cut_lines.append((x1, y1, x2, y2))

        connector_radius = (
            float(connector_radius) if connector_radius is not None else 2.5 * hinge_width
        )

        svg_lines: List[str] = []
        svg_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_lines.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'version="1.1" viewBox="0 0 {width_bb:.8f} {height_bb:.8f}">'
        )
        svg_lines.append('  <g id="cuts" fill="none" stroke-linecap="round" stroke-linejoin="round">')

        if cut_lines:
            svg_lines.append("    <!-- Tile outlines / full cuts -->")
            for x1, y1, x2, y2 in cut_lines:
                svg_lines.append(
                    f'    <line x1="{x1:.8f}" y1="{y1:.8f}" '
                    f'x2="{x2:.8f}" y2="{y2:.8f}" stroke="black" '
                    f'stroke-width="{cut_width:.8f}" />'
                )

        svg_lines.append("  </g>")

        if connector_points:
            svg_lines.append('  <g id="connectors" fill="red" stroke="none">')
            svg_lines.append("    <!-- Tiny connector dots at edge intersections (do NOT cut) -->")
            for pt in connector_points:
                x, y = _to_svg_xy(pt)
                svg_lines.append(
                    f'    <circle cx="{x:.8f}" cy="{y:.8f}" r="{connector_radius:.8f}" />'
                )
            svg_lines.append("  </g>")

        svg_lines.append("</svg>")

        os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
        with open(svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_lines))

        if not preview_svg_path:
            return

        # Preview with positive tiles over a negative-space background.
        tile_polys: List[str] = []
        for quad in self.structure.quads:
            poly_pts = [pts0[int(k)] for k in quad]
            pts_str = " ".join(f"{_to_svg_xy(p)[0]:.8f},{_to_svg_xy(p)[1]:.8f}" for p in poly_pts)
            tile_polys.append(pts_str)

        svg_preview: List[str] = []
        svg_preview.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_preview.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'version="1.1" viewBox="0 0 {width_bb:.8f} {height_bb:.8f}">'
        )
        svg_preview.append(
            '  <rect x="0" y="0" width="{:.8f}" height="{:.8f}" fill="#f0f4ff" />'.format(
                width_bb, height_bb
            )
        )

        if tile_polys:
            svg_preview.append(
                '  <g id="tiles" fill="#ffe8cc" stroke="black" stroke-width="{:.8f}" stroke-linejoin="round">'.format(
                    cut_width
                )
            )
            for pts_str in tile_polys:
                svg_preview.append(f'    <polygon points="{pts_str}" />')
            svg_preview.append("  </g>")

        if connector_points:
            svg_preview.append('  <g id="connectors" fill="#d33" stroke="none">')
            for pt in connector_points:
                x, y = _to_svg_xy(pt)
                svg_preview.append(
                    f'    <circle cx="{x:.8f}" cy="{y:.8f}" r="{connector_radius:.8f}" />'
                )
            svg_preview.append("  </g>")

        svg_preview.append("</svg>")
        os.makedirs(os.path.dirname(preview_svg_path) or ".", exist_ok=True)
        with open(preview_svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_preview))

    def export_dxf(
        self,
        dxf_path: str,
        *,
        phi_flat: float = np.pi,
        hinge_width: float = 0.01,
        phi_in_degrees: bool = False,
        connector_radius: Optional[float] = None,
        connector_radius_mm: Optional[float] = None,
        target_size_mm: float = 100.0,
    ) -> None:
        pts0, width_bb, height_bb, cut_segments, connector_points = self._layout_flat(
            phi_flat, phi_in_degrees=phi_in_degrees
        )

        longest_side = max(width_bb, height_bb, 1.0)
        scale = 1.0
        if target_size_mm is not None and target_size_mm > 0.0:
            scale = float(target_size_mm) / float(longest_side)

        def _scale_xy(p: np.ndarray) -> Tuple[float, float]:
            return float(p[0] * scale), float(p[1] * scale)

        cut_lines: List[Tuple[float, float, float, float]] = []
        for p0, p1 in cut_segments:
            x1, y1 = _scale_xy(p0)
            x2, y2 = _scale_xy(p1)
            cut_lines.append((x1, y1, x2, y2))

        base_connector_radius = (
            float(connector_radius) if connector_radius is not None else 2.5 * hinge_width
        )
        connector_radius_out = (
            float(connector_radius_mm)
            if connector_radius_mm is not None
            else base_connector_radius * scale
        )

        connector_circles: List[Tuple[float, float, float]] = []
        if connector_points and connector_radius_out > 0.0:
            for pt in connector_points:
                x, y = _scale_xy(pt)
                connector_circles.append((x, y, connector_radius_out))

        _write_basic_r12_dxf(cut_lines, connector_circles, out_path=dxf_path)


def export_edges_to_svg_grouped(
    points: np.ndarray,
    boundary_edges: Iterable[Tuple[int, int]],
    hinge_edges: Iterable[Tuple[int, int]],
    out_path: str,
    cut_width: float = 0.02,
    hinge_width: float = 0.01,
) -> None:
    """
    Export edges with separate styling for outer cuts vs internal hinges.

    - boundary_edges: interpreted as full-depth cuts (outer boundary)
    - hinge_edges:    interpreted as delicate hinge axes between tiles

    Geometry is kept exact; we only translate to [0, W]x[0, H] and flip y
    for SVG's coordinate convention. No scaling/shearing is applied.
    """

    pts0, width, height = _normalize_to_origin(points)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Degenerate pattern: zero area bounding box.")

    boundary_edges = list(boundary_edges)
    hinge_edges = list(hinge_edges)

    def _line_elems(e_iter: Iterable[Tuple[int, int]], stroke: float, color: str) -> List[str]:
        elems: List[str] = []
        for i, j in e_iter:
            p0 = pts0[int(i)]
            p1 = pts0[int(j)]
            # SVG y grows downward; flip using height
            x1, y1 = float(p0[0]), float(height - p0[1])
            x2, y2 = float(p1[0]), float(height - p1[1])
            elems.append(
                f'    <line x1="{x1:.8f}" y1="{y1:.8f}" '
                f'x2="{x2:.8f}" y2="{y2:.8f}" stroke="{color}" '
                f'stroke-width="{stroke:.8f}" />'
            )
        return elems

    svg: List[str] = []
    svg.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'version="1.1" viewBox="0 0 {width:.8f} {height:.8f}">'
    )
    svg.append('  <g fill="none" stroke-linecap="round" stroke-linejoin="round">')
    if boundary_edges:
        svg.append("    <!-- Outer cuts -->")
        svg.extend(_line_elems(boundary_edges, cut_width, "black"))
    if hinge_edges:
        svg.append("    <!-- Internal hinges (tiny) -->")
        svg.extend(_line_elems(hinge_edges, hinge_width, "red"))
    svg.append("  </g>")
    svg.append("</svg>")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def export_edges_to_svg(
    points: np.ndarray,
    edges: Iterable[Tuple[int, int]],
    out_path: str,
    stroke_width: float = 0.02,
) -> None:
    """
    Backwards-compatible wrapper: export all edges as identical cuts.

    For hinge-aware export, use `export_edges_to_svg_grouped` instead.
    """
    export_edges_to_svg_grouped(
        points,
        boundary_edges=edges,
        hinge_edges=[],
        out_path=out_path,
        cut_width=stroke_width,
        hinge_width=stroke_width,
    )


def export_obj_pattern_to_svg(
    obj_path: str,
    svg_path: str,
    cut_width: float = 0.02,
    hinge_width: float = 0.01,
    hinge_gap_fraction: float = 0.25,
    connector_radius: float = None,
    preview_svg_path: Optional[str] = None,
) -> None:
    """
    Read an OBJ pattern (points + quads) and export to SVG for laser cutting.

    - All tile edges are drawn as full cut lines.
    - Connector dots are placed at intersections of edges from *different* tiles
      (the four corners of every negative space, plus boundary openings).
    Args:
        obj_path:           Input OBJ with quad faces.
        svg_path:           Output SVG path.
        cut_width:          Stroke width for cuts.
        hinge_width:        Visual size for connectors (also used as default radius seed).
        hinge_gap_fraction: Ignored; kept for backward compatibility with older callers.
        connector_radius:   Optional absolute connector radius (if None, 2.5*hinge_width).
    """
    points, faces = read_obj(obj_path)
    if faces.ndim != 2 or faces.shape[1] != 4:
        raise ValueError("Expected quad faces (4 indices per face) in OBJ pattern.")
    faces = faces.astype(int) - 1  # OBJ indices are 1-based

    pts2d = np.asarray(points, dtype=float)[:, :2]
    pts0, width, height = _normalize_to_origin(pts2d)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Degenerate pattern: zero-area bounding box.")

    # Cut edges: unique geometric edges for laser cutting.
    cut_segments = _unique_segments_for_cuts(pts0, faces, scale=max(width, height, 1.0))

    # Connector defs built once from this static geometry; evaluated directly.
    connector_defs = _build_connector_defs_from_layout(pts0, faces)
    connector_points = _evaluate_connector_points(connector_defs, pts0, faces)

    def _to_svg_xy(p: np.ndarray) -> Tuple[float, float]:
        x = float(p[0])
        y = float(height - p[1])
        return x, y

    cut_lines: List[Tuple[float, float, float, float]] = []
    for p0, p1 in cut_segments:
        x1, y1 = _to_svg_xy(p0)
        x2, y2 = _to_svg_xy(p1)
        cut_lines.append((x1, y1, x2, y2))

    connector_radius = (
        float(connector_radius) if connector_radius is not None else 2.5 * hinge_width
    )

    svg_lines: List[str] = []
    svg_lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    svg_lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'version="1.1" viewBox="0 0 {width:.8f} {height:.8f}">'
    )

    svg_lines.append('  <g id="cuts" fill="none" stroke-linecap="round" stroke-linejoin="round">')
    if cut_lines:
        svg_lines.append("    <!-- Tile outlines / full cuts -->")
        for x1, y1, x2, y2 in cut_lines:
            svg_lines.append(
                f'    <line x1="{x1:.8f}" y1="{y1:.8f}" '
                f'x2="{x2:.8f}" y2="{y2:.8f}" stroke="black" '
                f'stroke-width="{cut_width:.8f}" />'
            )
    svg_lines.append("  </g>")

    if connector_points:
        svg_lines.append('  <g id="connectors" fill="red" stroke="none">')
        svg_lines.append("    <!-- Tiny connector dots at edge intersections (do NOT cut) -->")
        for pt in connector_points:
            x, y = _to_svg_xy(pt)
            svg_lines.append(f'    <circle cx="{x:.8f}" cy="{y:.8f}" r="{connector_radius:.8f}" />')
        svg_lines.append("  </g>")

    svg_lines.append("</svg>")

    os.makedirs(os.path.dirname(svg_path) or ".", exist_ok=True)
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_lines))

    # Optional preview with positive tiles over a negative-space background.
    if preview_svg_path:
        svg_preview: List[str] = []
        svg_preview.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_preview.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'version="1.1" viewBox="0 0 {width:.8f} {height:.8f}">'
        )
        svg_preview.append(
            '  <rect x="0" y="0" width="{:.8f}" height="{:.8f}" fill="#f0f4ff" />'.format(
                width, height
            )
        )

        # Positive tiles (fill), outlines, and connector dots.
        tile_polys: List[str] = []
        for quad in faces:
            poly_pts = [pts0[int(k)] for k in quad]
            pts_str = " ".join(f"{float(p[0]):.8f},{float(height - p[1]):.8f}" for p in poly_pts)
            tile_polys.append(pts_str)

        if tile_polys:
            svg_preview.append(
                '  <g id="tiles" fill="#ffe8cc" stroke="black" stroke-width="{:.8f}" stroke-linejoin="round">'.format(
                    cut_width
                )
            )
            for pts_str in tile_polys:
                svg_preview.append(f'    <polygon points="{pts_str}" />')
            svg_preview.append("  </g>")

        if connector_points:
            svg_preview.append('  <g id="connectors" fill="#d33" stroke="none">')
            for pt in connector_points:
                x, y = _to_svg_xy(pt)
                svg_preview.append(
                    f'    <circle cx="{x:.8f}" cy="{y:.8f}" r="{connector_radius:.8f}" />'
                )
            svg_preview.append("  </g>")

        svg_preview.append("</svg>")
        os.makedirs(os.path.dirname(preview_svg_path) or ".", exist_ok=True)
        with open(preview_svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(svg_preview))


def export_eps_pattern_to_svg(
    eps: np.ndarray,
    svg_path: str,
    phi_flat: float = np.pi,
    cut_width: float = 0.02,
    hinge_width: float = 0.01,
    hinge_gap_fraction: float = 0.25,
    phi_in_degrees: bool = False,
    connector_radius: float = None,
    preview_svg_path: Optional[str] = None,
    phi_ref: Optional[float] = None,
    phi_ref_in_degrees: bool = False,
) -> None:
    """
    Build a MatrixStructure from an interior-offset field eps and export its
    flat rectangular layout (at phi = phi_flat) to SVG with connector dots.

    - Edges are drawn as cuts.
    - Connector dots mark intersections of edges from different tiles (negative
      space corners); computed before they visually vanish in compact layouts.
    - hinge_gap_fraction is accepted for API compatibility but ignored (connectors
      are point-like now).
    - phi_flat can be provided in radians (default) or degrees if phi_in_degrees=True.
    - phi_ref (default 90 degrees) sets which layout defines connector positions so
      they persist even when phi_flat is 0 or pi.
    """
    exporter = EpsLaserExporter(eps, phi_ref=phi_ref, phi_ref_in_degrees=phi_ref_in_degrees)
    exporter.export_svg(
        svg_path,
        phi_flat=phi_flat,
        cut_width=cut_width,
        hinge_width=hinge_width,
        phi_in_degrees=phi_in_degrees,
        connector_radius=connector_radius,
        preview_svg_path=preview_svg_path,
    )


def export_eps_pattern_to_dxf(
    eps: np.ndarray,
    dxf_path: str,
    phi_flat: float = np.pi,
    hinge_width: float = 0.01,
    hinge_gap_fraction: float = 0.25,
    phi_in_degrees: bool = False,
    connector_radius: float = None,
    connector_radius_mm: float = None,
    target_size_mm: float = 100.0,
    phi_ref: Optional[float] = None,
    phi_ref_in_degrees: bool = False,
) -> None:
    """
    Export an eps field to a DXF with cut lines and connector dots.

    - CUT layer contains all edges (outer boundary + negative-space cuts).
    - CONNECTORS layer contains tiny circles at edge intersections; laser should
      ignore this layer or mark lightly so connectors are not cut through.
    - target_size_mm scales the longer dimension of the bounding box to a
      physical size (keeps aspect ratio, so the shorter side may be smaller).
    - connector_radius_mm overrides the scaled connector size to an absolute
      physical radius; otherwise the radius scales with the pattern.
    """
    exporter = EpsLaserExporter(eps, phi_ref=phi_ref, phi_ref_in_degrees=phi_ref_in_degrees)
    exporter.export_dxf(
        dxf_path,
        phi_flat=phi_flat,
        hinge_width=hinge_width,
        phi_in_degrees=phi_in_degrees,
        connector_radius=connector_radius,
        connector_radius_mm=connector_radius_mm,
        target_size_mm=target_size_mm,
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export kirigami patterns to a simple 2D SVG for laser cutting.\n"
            "Supports OBJ patterns (e.g. files from the `pattern/` folder)."
        )
    )
    parser.add_argument(
        "--obj",
        type=str,
        required=True,
        help="Input OBJ file containing a quad mesh pattern.",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output SVG file path.",
    )
    parser.add_argument(
        "--cut-width",
        type=float,
        default=0.02,
        help="Stroke width in model units for outer cuts (purely visual).",
    )
    parser.add_argument(
        "--hinge-width",
        type=float,
        default=0.01,
        help=("Size in model units for connector dots (use a tiny value to emphasize their role)."),
    )
    parser.add_argument(
        "--hinge-gap-frac",
        type=float,
        default=0.25,
        help="Accepted for backward compatibility; ignored in the connector-dot export.",
    )
    args = parser.parse_args()

    export_obj_pattern_to_svg(
        args.obj,
        args.out,
        cut_width=args.cut_width,
        hinge_width=args.hinge_width,
        hinge_gap_fraction=args.hinge_gap_frac,
    )


if __name__ == "__main__":
    _cli()
