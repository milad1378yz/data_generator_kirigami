import numpy as np
import matplotlib.pyplot as plt


# ---------- core 4-bar map (paper eq. (1)) ----------
def R(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def compute_linkage(x0: np.ndarray, x3: np.ndarray, phi: float, eps: float) -> np.ndarray:
    I = np.eye(2)
    Q = (1.0 + eps) * R(-phi)
    x1 = (I - Q) @ x0 + Q @ x3
    x2 = -Q @ x0 + (I + Q) @ x3
    return np.vstack([x0, x1, x2, x3])  # [x0,x1,x2,x3]


# ---------- helpers ----------
def checkerboard_phi(m, n, phi):
    ij = np.add.outer(np.arange(m), np.arange(n))
    return np.where(ij % 2 == 0, phi, np.pi - phi)


# March the whole m×n array, *one negative space at a time* (forward DP)
def march_array(m, n, phi_field, eps_field, top_seeds, left_seeds):
    """
    top_seeds[j]  = x_{0,j,3} for j=0..n-1  (top boundary seed nodes)
    left_seeds[i] = x_{i,0,0} for i=0..m-1  (left boundary seed nodes)
    Returns nodes[i,j,k,:] with k in {0,1,2,3}.
    """
    nodes = np.zeros((m, n, 4, 2), dtype=float)
    for i in range(m):  # row by row
        for j in range(n):  # left to right
            x0 = left_seeds[i] if j == 0 else nodes[i, j - 1, 2]  # share x2 from the left
            x3 = top_seeds[j] if i == 0 else nodes[i - 1, j, 1]  # share x1 from above
            nodes[i, j] = compute_linkage(x0, x3, phi_field[i, j], eps_field[i, j])
    return nodes


def draw_linkages(nodes):
    m, n = nodes.shape[:2]
    plt.figure(figsize=(7, 7))
    for i in range(m):
        for j in range(n):
            P = nodes[i, j]
            cyc = [0, 1, 2, 3, 0]
            plt.plot(P[cyc, 0], P[cyc, 1], lw=1)
            # optionally mark the four vertices:
            plt.plot(P[:, 0], P[:, 1], "o", ms=2)
    plt.axis("equal")
    plt.axis("off")
    plt.savefig("linkages.png", dpi=300)


# ---------- rectangular inverse (boundary → seeds) ----------
def seeds_from_rectangle(TL, TR, BL, m, n, *, mode="midpoint"):
    """
    Return:
      top_seeds: (n,2) positions along TL->TR
      left_seeds: (m,2) positions along TL->BL
      corners: dict(TL,TR,BL,BR)

    'mode="midpoint"' places seeds at segment midpoints, avoiding corner duplication.
    """
    TL = np.asarray(TL, float)
    TR = np.asarray(TR, float)
    BL = np.asarray(BL, float)
    top_vec = TR - TL
    left_vec = BL - TL

    if mode == "midpoint":
        t = (np.arange(n) + 0.5) / n  # n midpoints, exclude corners
        s = (np.arange(m) + 0.5) / m
    else:
        # small corner offsets if you prefer not to use midpoints
        eps = 1e-3
        t = np.linspace(eps, 1.0 - eps, n)
        s = np.linspace(eps, 1.0 - eps, m)

    top_seeds = TL[None, :] + t[:, None] * top_vec[None, :]
    left_seeds = TL[None, :] + s[:, None] * left_vec[None, :]
    BR = BL + (TR - TL)
    corners = dict(TL=TL, TR=TR, BL=BL, BR=BR)
    return top_seeds, left_seeds, corners


def _advance_on_edge(b_curr, P_adj, edge_vec, edge_p0, phi_g, eps_b, project_to_edge=True):
    """
    One ghost step from current boundary node b_curr using adjacent interior node P_adj,
    then optionally project result back to the edge line passing through edge_p0 with direction edge_vec.
    """
    I = np.eye(2)
    Q = (1.0 + eps_b) * R(-phi_g)

    # Build ghost linkage with seeds (x0, x3) = (b_curr, P_adj)
    # and pick the vertex that advances along the edge direction.
    x1 = (I - Q) @ b_curr + Q @ P_adj
    x2 = -Q @ b_curr + (I + Q) @ P_adj

    def score(x):
        v = x - b_curr
        fwd = np.dot(v, edge_vec) / (np.linalg.norm(edge_vec) ** 2 + 1e-12)
        # prefer small lateral drift as tie‑breaker
        u = x - edge_p0
        lat = np.linalg.norm(
            u - (np.dot(u, edge_vec) / (np.linalg.norm(edge_vec) ** 2 + 1e-12)) * edge_vec
        )
        return (fwd, -lat)

    x_next = x1 if score(x1) >= score(x2) else x2

    if project_to_edge:
        u = x_next - edge_p0
        t = np.dot(u, edge_vec) / (np.linalg.norm(edge_vec) ** 2 + 1e-12)
        x_next = edge_p0 + t * edge_vec
    return x_next


def lift_boundary_aligned(nodes, phi_field, corners, *, eps_b=0.0, project_to_edge=True):
    """
    Reconstruct bottom/right/top/left boundary nodes from interior using ghost four‑bar steps,
    then (optionally) project them to the rectangle edges so everything is aligned.
    """
    m, n = nodes.shape[:2]
    TL, TR, BL, BR = corners["TL"], corners["TR"], corners["BL"], corners["BR"]
    v_top, v_left = TR - TL, BL - TL
    v_right, v_bottom = BR - TR, BL - BR  # note: bottom marches right->left

    b_left = np.zeros((m + 1, 2))
    b_left[0] = TL
    b_top = np.zeros((n + 1, 2))
    b_top[0] = TL
    b_right = np.zeros((m + 1, 2))
    b_right[0] = TR
    b_bottom = np.zeros((n + 1, 2))
    b_bottom[0] = BR

    # Left: TL -> BL using adjacent x_{i,0,0}
    for i in range(m):
        phi_g = np.pi - phi_field[i, 0]
        b_left[i + 1] = _advance_on_edge(
            b_left[i], nodes[i, 0, 0], v_left, TL, phi_g, eps_b, project_to_edge
        )

    # Top: TL -> TR using adjacent x_{0,j,3}
    for j in range(n):
        phi_g = np.pi - phi_field[0, j]
        b_top[j + 1] = _advance_on_edge(
            b_top[j], nodes[0, j, 3], v_top, TL, phi_g, eps_b, project_to_edge
        )

    # Right: TR -> BR using adjacent x_{i,n-1,2}
    for i in range(m):
        phi_g = np.pi - phi_field[i, n - 1]
        b_right[i + 1] = _advance_on_edge(
            b_right[i], nodes[i, n - 1, 2], v_right, TR, phi_g, eps_b, project_to_edge
        )

    # Bottom: BR -> BL using adjacent x_{m-1, j,1} marching right->left
    for j in range(n):
        phi_g = np.pi - phi_field[m - 1, n - 1 - j]
        b_bottom[j + 1] = _advance_on_edge(
            b_bottom[j], nodes[m - 1, n - 1 - j, 1], v_bottom, BR, phi_g, eps_b, project_to_edge
        )
    b_bottom = b_bottom[::-1]  # put in BL -> BR order

    # pin the exact corners
    b_left[0], b_top[0] = TL, TL
    b_top[-1], b_right[0] = TR, TR
    b_right[-1], b_bottom[-1] = BR, BR
    b_bottom[0], b_left[-1] = BL, BL
    return dict(top=b_top, right=b_right, bottom=b_bottom, left=b_left)


# ---------- tiny driver: inverse-to-rectangle via marching ----------
def inverse_rectangle(
    m=4,
    n=6,
    phi=np.pi / 2,
    eps_field=None,
    TL=(0.0, 0.0),
    TR=(6.0, 0.0),
    BL=(0.0, -4.0),
    eps_b=0.0,
    plot=True,
):
    if eps_field is None:
        eps_field = np.zeros((m, n), float)  # interior offsets are free design vars
    phi_field = checkerboard_phi(m, n, phi)

    # seeds from target rectangle (identity M_sub choice)
    top_seeds, left_seeds, corners = seeds_from_rectangle(TL, TR, BL, m, n)

    # march interior
    nodes = march_array(m, n, phi_field, eps_field, top_seeds, left_seeds)

    # plot the nodes if desired
    if plot:
        draw_linkages(nodes)

    # lift boundary (ghost four-bar)
    boundary = lift_boundary_aligned(nodes, phi_field, corners, eps_b=eps_b, project_to_edge=True)

    if plot:
        plt.figure(figsize=(7, 7))
        # draw linkages
        for i in range(m):
            for j in range(n):
                P = nodes[i, j]
                cyc = [0, 1, 2, 3, 0]
                plt.plot(P[cyc, 0], P[cyc, 1], lw=1)
        # draw rectangle target
        TL, TR, BL, BR = corners["TL"], corners["TR"], corners["BL"], corners["BR"]
        rect = np.vstack([TL, TR, BR, BL, TL])
        plt.plot(rect[:, 0], rect[:, 1], linestyle="--", linewidth=2)

        # draw recovered boundary
        for key in ["top", "right", "bottom", "left"]:
            B = boundary[key]
            plt.plot(B[:, 0], B[:, 1], linewidth=2)

        plt.axis("equal")
        plt.axis("off")
        plt.title("Inverse to Rectangle (marching + boundary lift)")
        plt.show()

    return nodes, boundary, corners


# # ---------- example usage ----------
# m = n = 3
# phi_global = np.pi / 2

# # interior offsets (your example) — tune these to sculpt the reconfigured shape
# eps = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)

# # choose two boundary *seed* polylines instead of "+2" hacks
# # (edit these three points to set your shape)
# x00 = np.array([0.0, 0.0])  # left seed for (0,0): x_{0,0,0}
# x03 = np.array([1.0, 1.0])  # top  seed for (0,0): x_{0,0,3}
# top_right = x03 + np.array([4.0, 0.0])  # end of the top seed line
# bottom_left = x00 + np.array([0.0, -4.0])  # end of the left seed line

# # sample the seeds along those lines (replace with any polyline sampler you like)
# top_seeds = linspace2d(x03, top_right, n)  # x_{0,j,3}, j=0..n-1
# left_seeds = linspace2d(x00, bottom_left, m)  # x_{i,0,0}, i=0..m-1

# phi_field = checkerboard_phi(m, n, phi_global)  # compact-reconfigurable field

# nodes = march_array(m, n, phi_field, eps, top_seeds, left_seeds)
# draw_linkages(nodes)


# make a 4×6 array that deploys with all rectangular slits (phi=pi/2) to a 6×4 rectangle
nodes, boundary, corners = inverse_rectangle(
    m=3, n=3, phi=np.pi / 2, eps_b=0.0, TL=(0.0, 0.0), TR=(4.0, 0.0), BL=(0.0, -4.0)
)
