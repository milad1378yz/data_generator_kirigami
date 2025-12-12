import numpy as np


def norm(x):
    """
    Calculate the Euclidean norm (magnitude) of a vector.

    Args:
        x (array-like): Input vector of any dimension

    Returns:
        float: The Euclidean norm of the vector

    Shape transition: (n,) -> scalar
    """
    return np.linalg.norm(x)


def normalize(x):
    """
    Normalize a vector to unit length.

    Args:
        x (array-like): Input vector to normalize

    Returns:
        ndarray: Unit vector in the same direction as input

    Shape transition: (n,) -> (n,) with norm = 1
    """
    return x / norm(x)


def is_even(x):
    """
    Check if a number is even using modular arithmetic.

    Args:
        x (int): Integer to check

    Returns:
        bool: True if x is even, False otherwise

    Note: Uses (x + 1) % 2 which returns True for even numbers
    """
    return bool((x + 1) % 2)


def is_odd(x):
    """
    Check if a number is odd using modular arithmetic.

    Args:
        x (int): Integer to check

    Returns:
        bool: True if x is odd, False otherwise

    Note: Uses x % 2 which returns True for odd numbers
    """
    return bool(x % 2)


def cyclic(x, a):
    """
    Cyclically shift array elements along the first axis.

    Args:
        x (ndarray): Input array to shift
        a (int): Number of positions to shift (positive = right shift)

    Returns:
        ndarray: Array with cyclically shifted elements

    Shape transition: (n, ...) -> (n, ...) with elements reordered
    """
    return np.roll(x, a, axis=0)

def empty_list_of_lists(n):
    """
    Create a list of n empty lists.

    Args:
        n (int): Number of empty lists to create

    Returns:
        list: List containing n empty lists

    Shape creation: Creates structure for n independent sublists
    """
    return [[] for _ in range(n)]


def rotation_matrix(angle):
    """
    Create a 2D rotation matrix for given angle.

    Args:
        angle (float): Rotation angle in radians

    Returns:
        ndarray: 2x2 rotation matrix

    Shape: (2, 2)
    Matrix form: [[cos(θ), -sin(θ)],
                  [sin(θ),  cos(θ)]]
    """
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotation_matrix_3d(angle):
    """
    Create a 3D rotation matrix for rotation around z-axis.

    Args:
        angle (float): Rotation angle in radians around z-axis

    Returns:
        ndarray: 3x3 rotation matrix

    Shape: (3, 3)
    Matrix form: [[cos(θ), -sin(θ), 0],
                  [sin(θ),  cos(θ), 0],
                  [0,       0,      1]]
    """
    return np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )


def rotation_matrix_homog(angle):
    """
    Create a 3*3 homogeneous rotation matrix for 2D rotation.

    Args:
        angle (float): Rotation angle in radians

    Returns:
        ndarray: 3*3 homogeneous transformation matrix

    Shape: (3, 3)
    Matrix form: [[cos(θ), -sin(θ), 0],
                  [sin(θ),  cos(θ), 0],
                  [0,       0,      1]]
    """
    zero_row = np.array([0.0, 0.0])
    homog_col = np.array([[0.0], [0.0], [1.0]])
    return np.hstack([np.vstack([rotation_matrix(angle), zero_row]), homog_col])


def translation_matrix_homog(tx, ty):
    """
    Create a 3*3 homogeneous translation matrix for 2D translation.

    Args:
        tx (float): Translation in x-direction
        ty (float): Translation in y-direction

    Returns:
        ndarray: 3*3 homogeneous transformation matrix

    Shape: (3, 3)
    Matrix form: [[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]]
    """
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def multiply_matrices(mats):
    """
    Multiply a list of matrices from left to right using recursion.

    Args:
        mats (list): List of matrices to multiply

    Returns:
        ndarray: Product of all matrices

    Shape transition: Depends on input matrices, follows matrix multiplication rules
    Note: Modifies input list by consuming elements during recursion
    """
    if len(mats) == 2:
        return np.dot(mats[0], mats[1])
    else:
        mats[-2] = np.dot(mats[-2], mats[-1])
        mats.pop()
        return multiply_matrices(mats)


def rotate_points(points, origin, angle):
    """
    Rotate points around a given origin by specified angle.

    Args:
        points (ndarray): Points to rotate, shape (n, 2) or (n, 3) or (2,) or (3,)
        origin (array-like): Center of rotation, shape (2,) or (3,)
        angle (float): Rotation angle in radians

    Returns:
        ndarray: Rotated points with same shape as input

    Shape transition:
        - Input: (n, d) where d=2 or 3, or (d,)
        - Output: Same shape as input

    Process:
        1. Translate points to origin
        2. Apply rotation matrix
        3. Translate back to original position
    """

    points = np.asarray(points, dtype=float)
    origin = np.asarray(origin, dtype=float)

    if points.ndim == 1:
        onedim = True
        points_2d = points[None, :]
    else:
        onedim = False
        points_2d = points

    if points_2d.shape[1] == 3:
        rot_mat = rotation_matrix_3d(angle)
    else:
        rot_mat = rotation_matrix(angle)

    rotated_points = (points_2d - origin) @ rot_mat.T + origin
    return rotated_points[0] if onedim else rotated_points


def planar_cross(a, b):
    """
    Calculate the 2D cross product (determinant) of two 2D vectors.

    Args:
        a (array-like): First 2D vector, shape (2,)
        b (array-like): Second 2D vector, shape (2,)

    Returns:
        float: Cross product a[0]*b[1] - a[1]*b[0]

    Shape transition: (2,) * (2,) -> scalar
    Geometric interpretation: Magnitude of z-component of 3D cross product
    """

    return a[0] * b[1] - a[1] * b[0]


def calculate_angle(a, b, c):
    """
    Calculate the angle ABC (angle at point B) in the range [0, 2π).

    Args:
        a (array-like): First point, shape (2,)
        b (array-like): Vertex point (angle measured here), shape (2,)
        c (array-like): Third point, shape (2,)

    Returns:
        float: Angle ABC in radians, range [0, 2π)

    Process:
        1. Create unit vectors from B to A and B to C
        2. Use atan2 of cross and dot products for full angular range
        3. Normalize to [0, 2π) range
    """

    a = np.array(a, copy=True)
    b = np.array(b, copy=True)
    c = np.array(c, copy=True)

    ab_hat = normalize(b - a)
    ac_hat = normalize(c - a)

    x = np.dot(ab_hat, ac_hat)
    y = planar_cross(ab_hat, ac_hat)

    atan2_angle = np.arctan2(y, x)

    return atan2_angle % (2.0 * np.pi)


def shift_points(points, shift):
    """
    Translate all points by a constant shift vector.

    Args:
        points (ndarray): Points to translate, shape (n, d)
        shift (array-like): Translation vector, shape (d,)

    Returns:
        ndarray: Translated points, shape (n, d)

    Shape transition: (n, d) + (d,) -> (n, d)
    Operation: Each point[i] += shift
    """

    return points + shift


def plot_structure(points, quads, linkages, ax):
    """
    Plot a kirigami structure showing quadrilateral faces.

    Args:
        points (ndarray): Vertex coordinates, shape (n, 2)
        quads (ndarray): Quad face indices, shape (m, 4)
        linkages (ndarray): Linkage connectivity (unused in current implementation)
        ax (matplotlib.axes.Axes): Matplotlib axes object for plotting

    Shape requirements:
        - points: (n_vertices, 2) for 2D coordinates
        - quads: (n_quads, 4) for quad vertex indices

    Visual styling:
        - Fill color: Light peach (1, 229/255, 204/255)
        - Edge color: Black
        - Edge width: 2
        - Alpha: 0.8
    """

    for i, quad in enumerate(quads):
        x = points[quad, 0]
        y = points[quad, 1]
        ax.fill(x, y, color=(1, 229 / 255, 204 / 255), edgecolor="k", linewidth=2, alpha=0.8)

    ax.axis("off")
    ax.set_aspect("equal")


def _orientation2d(a, b, c):
    """Return z-component of the 2D cross product for orientation tests."""
    return np.cross(b - a, c - a)


def _point_on_segment(a, b, c, tol=1e-9):
    """Check whether point b lies on segment [a, c], with tolerance."""
    return (
        min(a[0], c[0]) - tol <= b[0] <= max(a[0], c[0]) + tol
        and min(a[1], c[1]) - tol <= b[1] <= max(a[1], c[1]) + tol
    )


def _segments_intersect2d(p0, p1, p2, p3, tol=1e-9):
    """Return True if segments [p0, p1] and [p2, p3] intersect."""
    o1 = _orientation2d(p0, p1, p2)
    o2 = _orientation2d(p0, p1, p3)
    o3 = _orientation2d(p2, p3, p0)
    o4 = _orientation2d(p2, p3, p1)

    if abs(o1) <= tol and _point_on_segment(p0, p2, p1, tol):
        return True
    if abs(o2) <= tol and _point_on_segment(p0, p3, p1, tol):
        return True
    if abs(o3) <= tol and _point_on_segment(p2, p0, p3, tol):
        return True
    if abs(o4) <= tol and _point_on_segment(p2, p1, p3, tol):
        return True

    return (o1 * o2 < -tol**2) and (o3 * o4 < -tol**2)


def find_invalid_quads(points, quads, area_tol=1e-8, tol=1e-9):
    """
    Identify quads that collapse or self-intersect.

    Args:
        points (ndarray): Vertex coordinates, shape (n, 2)
        quads (ndarray): Quad indices, shape (m, 4)
        area_tol (float): Minimum absolute area before flagging as collapsed
        tol (float): Tolerance for intersection tests

    Returns:
        list[tuple[int, str]]: (quad_index, reason) for each invalid quad.
    """
    invalid = []
    for idx, quad in enumerate(quads):
        poly = points[np.asarray(quad)]
        if not np.all(np.isfinite(poly)):
            invalid.append((idx, "non-finite coordinates"))
            continue

        area = 0.5 * sum(
            poly[i, 0] * poly[(i + 1) % 4, 1] - poly[(i + 1) % 4, 0] * poly[i, 1]
            for i in range(4)
        )
        if abs(area) <= area_tol:
            invalid.append((idx, "collapsed area"))
            continue

        if _segments_intersect2d(poly[0], poly[1], poly[2], poly[3], tol):
            invalid.append((idx, "self-intersection (01×23)"))
            continue

        if _segments_intersect2d(poly[1], poly[2], poly[3], poly[0], tol):
            invalid.append((idx, "self-intersection (12×30)"))
            continue

    return invalid


def _quad_edges(quad):
    """Return list of directed edge index pairs for a quad of 4 vertex indices."""
    return [
        (int(quad[0]), int(quad[1])),
        (int(quad[1]), int(quad[2])),
        (int(quad[2]), int(quad[3])),
        (int(quad[3]), int(quad[0])),
    ]


def _aabb_for_poly(poly):
    x = poly[:, 0]
    y = poly[:, 1]
    return (x.min(), x.max(), y.min(), y.max())


def _aabb_overlap(b1, b2, tol=0.0):
    return not (
        b1[1] < b2[0] - tol
        or b2[1] < b1[0] - tol
        or b1[3] < b2[2] - tol
        or b2[3] < b1[2] - tol
    )


def _point_in_polygon(pt, poly, tol=1e-9):
    """
    Ray casting point-in-polygon. Returns True if inside or on boundary.
    poly: (N,2) array; pt: (2,) array.
    """
    x, y = float(pt[0]), float(pt[1])
    n = len(poly)
    inside = False
    for i in range(n):
        j = (i - 1) % n
        xi, yi = float(poly[i, 0]), float(poly[i, 1])
        xj, yj = float(poly[j, 0]), float(poly[j, 1])

        # On-edge check
        if _point_on_segment(np.array([xj, yj]), np.array([x, y]), np.array([xi, yi]), tol):
            return True

        intersect = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-300) + xi
        )
        if intersect:
            inside = not inside
    return inside


def find_overlapping_quads(points, quads, tol=1e-9):
    """
    Detect overlaps between distinct quads (edge crossings or containment).

    This catches cases where individual quads are valid (non-self-intersecting)
    but different quads overlap each other, which can happen with large offsets.

    Returns a list of (i, j, reason) with i<j.
    """
    pts = np.asarray(points, dtype=float)
    qs = np.asarray(quads, dtype=int)
    n = len(qs)

    # Precompute per-quad data
    polys = [pts[q] for q in qs]
    bboxes = [_aabb_for_poly(poly) for poly in polys]
    edges = [_quad_edges(q) for q in qs]

    overlaps = []
    for i in range(n):
        qi = qs[i]
        poly_i = polys[i]
        box_i = bboxes[i]
        edge_i = edges[i]
        for j in range(i + 1, n):
            qj = qs[j]
            shared = set(qi.tolist()).intersection(set(qj.tolist()))
            # Skip pairs that share an entire edge (adjacent quads)
            if len(shared) >= 2:
                continue

            poly_j = polys[j]
            box_j = bboxes[j]
            if not _aabb_overlap(box_i, box_j, tol):
                continue

            # Edge-edge intersections (exclude edges that share endpoints)
            hit = False
            for a0, a1 in edge_i:
                p0, p1 = pts[a0], pts[a1]
                for b0, b1 in edges[j]:
                    if a0 in shared or a1 in shared or b0 in shared or b1 in shared:
                        # allow touching at shared vertices
                        continue
                    p2, p3 = pts[b0], pts[b1]
                    if _segments_intersect2d(p0, p1, p2, p3, tol):
                        overlaps.append((i, j, "edge crossing"))
                        hit = True
                        break
                if hit:
                    break
            if hit:
                continue

            # Containment: one polygon entirely inside the other
            # Pick a vertex not shared
            vi = next((k for k in qi if k not in shared), qi[0])
            vj = next((k for k in qj if k not in shared), qj[0])
            if _point_in_polygon(pts[vi], poly_j, tol) or _point_in_polygon(pts[vj], poly_i, tol):
                overlaps.append((i, j, "containment"))

    return overlaps


def read_obj(filename):
    """
    Read geometry data from Wavefront OBJ file format.

    Args:
        filename (str): Input filename

    Returns:
        tuple: (points, faces) where
            - points: ndarray of shape (n, 2) with vertex coordinates
            - faces: ndarray of shape (m, 4) with face indices (0-based)

    Parsing:
        - Lines starting with 'v': Vertex coordinates
        - Lines starting with 'f': Face indices (converted from 1-based to 0-based)
        - Other lines: Ignored
    """

    obj = open(filename, "r")

    points = []
    faces = []

    for line in obj:

        first_char = line[0]

        if first_char == "v":
            point = [float(_) for _ in line.split(" ")[1:-1]]
            points.append(point)

        elif first_char == "f":
            face = [int(_) for _ in line.replace("//", "").split(" ")[1:]]
            faces.append(face)

        else:
            continue

    obj.close()

    return np.array(points), np.array(faces)
