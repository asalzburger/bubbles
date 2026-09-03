import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, disk
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, convolve
from collections import deque, defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe
from pathlib import Path
import argparse
from scipy.ndimage import median_filter


# ──────────────────────────────────────────────
# 1. YOUR FUNCTIONS (unchanged, with minor fixes)
# ──────────────────────────────────────────────

def hoshen_kopelman(occupied, connectivity=8):
    m, n = occupied.shape
    parent = np.arange(m * n)
    rank   = np.zeros(m * n)
    labels = np.zeros((m, n), dtype=np.int32)
    next_label = 1
    offsets = [(-1,0),(0,-1),(-1,-1),(-1,1)] if connectivity == 8 else [(-1,0),(0,-1)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb: return ra
        if rank[ra] < rank[rb]: ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]: rank[ra] += 1
        return ra

    for i in range(m):
        for j in range(n):
            if not occupied[i, j]: continue
            nbs = []
            for di, dj in offsets:
                ni, nj = i+di, j+dj
                if 0 <= ni < m and 0 <= nj < n and labels[ni,nj] > 0:
                    nbs.append(labels[ni,nj]-1)
            if not nbs:
                labels[i,j] = next_label; next_label += 1
            else:
                root = nbs[0]
                for nb in nbs[1:]:
                    root = union(root, nb)
                labels[i,j] = root + 1

    for i in range(m):
        for j in range(n):
            if labels[i,j] > 0:
                labels[i,j] = find(labels[i,j]-1) + 1
    return labels


def prune_spurs(skel, max_spur_length=5):
    skel = skel.copy()
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    for _ in range(max_spur_length):
        neighbor_count = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
        endpoints = skel & (neighbor_count == 1)
        if not endpoints.any():
            break
        skel[endpoints] = False
    return skel


def trace_skeleton_clean(skel):
    ys, xs = np.where(skel)
    points = set(zip(ys.tolist(), xs.tolist()))
    if len(points) < 3:
        return [np.array(list(points), dtype=np.int32)] if points else []

    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    neighbor_cache = {}

    def neighbors(p):
        if p not in neighbor_cache:
            r, c = p
            neighbor_cache[p] = [(r+dr, c+dc) for dr, dc in DIRS if (r+dr, c+dc) in points]
        return neighbor_cache[p]

    endpoints, branch_points = [], []
    for p in points:
        nd = len(neighbors(p))
        if nd == 1:
            endpoints.append(p)
        elif nd >= 3:
            branch_points.append(p)

    nodes = set(endpoints + branch_points)

    def trace_chain(start, first_step):
        path = [start, first_step]
        visited = {start, first_step}
        cur = first_step
        while cur not in nodes:
            nbs = [nb for nb in neighbors(cur) if nb not in visited]
            if not nbs:
                break
            cur = nbs[0]
            path.append(cur)
            visited.add(cur)
        return path, cur

    edges = []
    seen = set()
    for node in nodes:
        for nb in neighbors(node):
            if nb in nodes:
                key = frozenset([node, nb])
                if key not in seen:
                    seen.add(key)
                    edges.append((node, nb, [node, nb]))
            else:
                path, end_node = trace_chain(node, nb)
                key = frozenset([node, end_node])
                if key not in seen:
                    seen.add(key)
                    edges.append((node, end_node, path))

    def edge_dir(edge, at_node):
        a, b, path = edge
        if at_node == a:
            return (path[1][0]-path[0][0], path[1][1]-path[0][1])
        else:
            return (path[-2][0]-path[-1][0], path[-2][1]-path[-1][1])

    node_edges = defaultdict(list)
    for i, (a, b, path) in enumerate(edges):
        node_edges[a].append(i)
        node_edges[b].append(i)

    used = set()
    trajectories = []

    for bp in branch_points:
        eidxs = node_edges[bp]
        dirs = [edge_dir(edges[i], bp) for i in eidxs]
        paired = [False] * len(eidxs)
        for i in range(len(eidxs)):
            if paired[i]: continue
            best_j, best_dot = -1, 2.0
            for j in range(len(eidxs)):
                if j == i or paired[j]: continue
                dot = dirs[i][0]*dirs[j][0] + dirs[i][1]*dirs[j][1]
                if dot < best_dot:
                    best_dot = dot; best_j = j
            if best_j >= 0 and best_dot < 0:
                paired[i] = paired[best_j] = True
                a_i, b_i, path_i = edges[eidxs[i]]
                a_j, b_j, path_j = edges[eidxs[best_j]]
                seg_i = path_i if bp == b_i else path_i[::-1]
                seg_j = path_j if bp == a_j else path_j[::-1]
                full = seg_i + seg_j[1:]
                trajectories.append(np.array(full, dtype=np.int32))
                used.add(eidxs[i]); used.add(eidxs[best_j])

        for i in range(len(eidxs)):
            if not paired[i] and eidxs[i] not in used:
                a, b, path = edges[eidxs[i]]
                seg = path if bp == b else path[::-1]
                if len(seg) > 2:
                    trajectories.append(np.array(seg, dtype=np.int32))
                used.add(eidxs[i])

    for i, (a, b, path) in enumerate(edges):
        if i not in used:
            trajectories.append(np.array(path, dtype=np.int32))
            used.add(i)

    return [t for t in trajectories if len(t) > 3]


# ──────────────────────────────────────────────
# 2. NOISY-IMAGE ROBUST PIPELINE
# ──────────────────────────────────────────────

def robust_extract_and_plot(input_path,
                            gap_bridge=5,       # max gap (px) to bridge in tracks
                            noise_size=15,      # max blob size to consider "noise"
                            min_track_len=40,   # min skeleton length for a valid track
                            max_spur=5,
                            elongation_ratio=2.5):
    """
    Full pipeline robust to:
      - Random saturated noise pixels / small blobs
      - Tracks with small gaps (disconnected fragments)
    """
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)

    # --- A. Threshold ---
    mx = arr.max(axis=2).astype(float)
    mn = arr.min(axis=2).astype(float)
    sat = np.where(mx > 0, (mx - mn) / mx, 0)
    fg = sat > threshold_otsu(sat[sat > 0])

    # --- B. Denoise: remove isolated specks BEFORE closing ---
    #     binary_opening kills 1-2 px isolated pixels
    fg = binary_closing(fg, disk(1))  # fill 1-px holes inside tracks
    fg = remove_small_objects(fg, min_size=noise_size)

    # --- C. BRIDGE GAPS: morphological closing with larger disk ---
    #     This is the KEY step for disconnected tracks.
    #     Dilation connects nearby fragments; erosion restores width.
    fg = binary_closing(fg, disk(gap_bridge))

    #     Re-remove noise that the closing may have merged into small blobs
    fg = remove_small_objects(fg, min_size=noise_size * 2)

    # --- D. CCL ---
    labels = hoshen_kopelman(fg, connectivity=8)
    n_comp = labels.max()
    print(f"Components after denoising + gap bridging: {n_comp}")

    # --- E. Per-component: skeletonize → prune → trace → FILTER ---
    all_trajs = []
    for c in range(1, n_comp + 1):
        comp = (labels == c)
        if comp.sum() < noise_size * 2:
            continue

        skel = skeletonize(comp)
        skel = prune_spurs(skel, max_spur_length=max_spur)

        # Quick length check before expensive tracing
        if skel.sum() < min_track_len:
            print(f"  Component {c}: skipped (skeleton too short: {skel.sum()} px)")
            continue

        paths = trace_skeleton_clean(skel)

        # --- F. ELONGATION FILTER: reject round blobs (noise that survived) ---
        for p in paths:
            if len(p) < min_track_len:
                continue
            # Bounding box elongation
            rows = p[:, 0]; cols = p[:, 1]
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1
            elongation = max(h, w) / max(min(h, w), 1)
            if elongation < elongation_ratio:
                continue  # too round → likely noise, not a track
            all_trajs.append((c, p))

    print(f"Valid tracks: {len(all_trajs)}")

    # --- G. Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Original (noisy)")

    axes[1].imshow(fg, cmap='gray')
    axes[1].set_title(f"After denoise + gap bridge (r={gap_bridge})")

    axes[2].imshow(img)
    for c, path in all_trajs:
        color = cm.tab20((c - 1) % 20 / 20)
        axes[2].plot(path[:, 1], path[:, 0], lw=2.5, color=color, alpha=0.9)
        axes[2].plot(path[0,1], path[0,0], 'o', color='lime', ms=6, zorder=5)
        axes[2].plot(path[-1,1], path[-1,0], 's', color='red', ms=6, zorder=5)
    axes[2].set_title(f"{len(all_trajs)} tracks detected")

    plt.tight_layout()
    plt.savefig("noisy_pipeline_result.png", dpi=150, bbox_inches='tight')
    plt.show()

    return all_trajs

def robust_extract_simulated(input_path,
                              gap_bridge=3,
                              noise_size=5,        # ← much lower for thin lines
                              min_track_len=10,    # ← allow short tracks
                              max_spur=3,
                              elongation_ratio=1.2):  # ← allow curved/short
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)

    # --- Threshold: use a LOWER threshold for thin colored lines on white ---
    mx = arr.max(axis=2).astype(float)
    mn = arr.min(axis=2).astype(float)
    sat = np.where(mx > 0, (mx - mn) / mx, 0)

    # For thin lines, Otsu on the full histogram can be dominated by the
    # white background. Use a fixed low threshold instead:
    fg = sat > 0.15  # ← much lower than 0.3; catches anti-aliased edges

    # --- Denoise: minimal for this type of image ---
    fg = remove_small_objects(fg, min_size=noise_size)

    # --- Bridge small gaps ---
    fg = binary_closing(fg, disk(gap_bridge))
    fg = remove_small_objects(fg, min_size=noise_size)

    # --- CCL ---
    labels = hoshen_kopelman(fg, connectivity=8)
    n_comp = labels.max()
    print(f"Components: {n_comp}")

    all_trajs = []
    for c in range(1, n_comp + 1):
        comp = (labels == c)
        if comp.sum() < noise_size:
            continue

        skel = skeletonize(comp)
        skel = prune_spurs(skel, max_spur_length=max_spur)

        if skel.sum() < min_track_len:
            continue

        paths = trace_skeleton_clean(skel)

        for p in paths:
            if len(p) < min_track_len:
                continue
            # Relaxed elongation check
            rows = p[:, 0]; cols = p[:, 1]
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1
            elongation = max(h, w) / max(min(h, w), 1)
            if elongation < elongation_ratio:
                # For very short tracks, skip the elongation check
                if len(p) < 20:
                    pass  # keep it
                else:
                    continue
            all_trajs.append((c, p))

    print(f"Valid tracks: {len(all_trajs)}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img)
    axes[0].set_title("Input")

    axes[1].imshow(img)
    for c, path in all_trajs:
        color = cm.tab20((c - 1) % 20 / 20)
        axes[1].plot(path[:, 1], path[:, 0], lw=2.5, color=color, alpha=0.9)
        axes[1].plot(path[0,1], path[0,0], 'o', color='lime', ms=6, zorder=5)
        axes[1].plot(path[-1,1], path[-1,0], 's', color='red', ms=6, zorder=5)
    axes[1].set_title(f"{len(all_trajs)} tracks")
    plt.tight_layout()
    plt.savefig("simulated_result.png", dpi=150, bbox_inches='tight')
    plt.show()
    return all_trajs   

# ──────────────────────────────────────────────
# 3. DEMO: Generate a synthetic noisy image
# ──────────────────────────────────────────────

def make_noisy_bubble_chamber(path="noisy_bubble.png", size=(800, 800), seed=42):
    """
    Creates a synthetic image with:
      - 4 curved "tracks" (arcs) with intentional gaps
      - Random saturated noise pixels scattered across the image
      - A few small noise blobs (3-10 px)
    """
    rng = np.random.default_rng(seed)
    h, w = size
    img = np.ones((h, w, 3), dtype=np.uint8) * 240  # light background

    # Draw 4 arc tracks (thick, colored)
    track_params = [
        (300, 400, 200, 30, 200, "blue"),     # (cx, cy, r, start_deg, end_deg, color)
        (400, 350, 150, 180, 330, "purple"),
        (250, 500, 180, 0, 150, "darkblue"),
        (500, 300, 120, 90, 270, "indigo"),
    ]

    for cx, cy, r, a0, a1, color in track_params:
        angles = np.linspace(np.radians(a0), np.radians(a1), 500)
        xs = (cx + r * np.cos(angles)).astype(int)
        ys = (cy + r * np.sin(angles)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs, ys = xs[valid], ys[valid]

        # Draw thick track (3 px wide)
        for x, y in zip(xs, ys):
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= y+dy < h and 0 <= x+dx < w:
                        img[y+dy, x+dx] = [40, 40, 200] if "blue" in color else [100, 20, 150]

        # INTENTIONALLY CREATE GAPS: erase random segments
        n_gaps = rng.integers(2, 5)
        for _ in range(n_gaps):
            gap_start = rng.integers(0, len(xs) - 20)
            gap_len = rng.integers(5, 15)  # 5-15 px gaps
            for idx in range(gap_start, min(gap_start + gap_len, len(xs))):
                x, y = xs[idx], ys[idx]
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if 0 <= y+dy < h and 0 <= x+dx < w:
                            img[y+dy, x+dx] = 240  # erase to background

    # Add random saturated noise pixels
    n_noise_px = 2000
    noise_ys = rng.integers(0, h, n_noise_px)
    noise_xs = rng.integers(0, w, n_noise_px)
    img[noise_ys, noise_xs] = rng.integers(100, 255, (n_noise_px, 3)).astype(np.uint8)

    # Add small noise blobs (3-10 px diameter)
    n_blobs = 30
    for _ in range(n_blobs):
        by = rng.integers(10, h-10)
        bx = rng.integers(10, w-10)
        br = rng.integers(2, 5)
        yy, xx = np.ogrid[-br:br+1, -br:br+1]
        mask = xx**2 + yy**2 <= br**2
        for dy, dx in np.argwhere(mask):
            if 0 <= by+dy < h and 0 <= bx+dx < w:
                img[by+dy, bx+dx] = rng.integers(100, 255, 3).astype(np.uint8)

    Image.fromarray(img).save(path)
    print(f"Saved noisy image: {path}")
    return path

def extract_from_bebch(input_path):
    img = Image.open(input_path).convert("L")
    gray = np.array(img, dtype=float)

    # --- Preprocess ---
    gray = median_filter(gray, size=3)
    fg = gray < threshold_otsu(gray)
    fg = remove_small_objects(fg, min_size=20)
    fg = binary_closing(fg, disk(2))

    # --- CCL ---
    labels = hoshen_kopelman(fg, connectivity=8)
    n_comp = labels.max()
    print(f"Total components: {n_comp}")

    spirals = []
    boundary = None
    track_fragments = []

    for c in range(1, n_comp + 1):
        comp = (labels == c)
        area = comp.sum()

        # Chamber boundary: largest component
        if area > 0.05 * fg.size:
            boundary = c
            continue

        skel = skeletonize(comp)
        skel = prune_spurs(skel, max_spur_length=3)
        if skel.sum() < 15:
            continue

        paths = trace_skeleton_clean(skel)
        for p in paths:
            if len(p) < 15:
                continue
            rows, cols = p[:, 0], p[:, 1]
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1
            elongation = max(h, w) / max(min(h, w), 1)

            # SPIRAL DETECTION: compact, not elongated, medium size
            if elongation < 1.5 and 30 < len(p) < 500:
                spirals.append((c, p))
            # Track fragment: elongated
            elif elongation > 2.0:
                track_fragments.append((c, p))

    print(f"Spirals: {len(spirals)}")
    print(f"Track fragments: {len(track_fragments)}")
    print(f"Boundary: {'detected' if boundary else 'not found'}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title("Original")

    axes[1].imshow(img, cmap='gray')
    for c, p in spirals:
        axes[1].plot(p[:, 1], p[:, 0], 'r-', lw=2, alpha=0.8)
    for c, p in track_fragments:
        axes[1].plot(p[:, 1], p[:, 0], 'b-', lw=1, alpha=0.4)
    axes[1].set_title(f"{len(spirals)} spirals (red), {len(track_fragments)} fragments (blue)")
    plt.tight_layout()
    plt.show()

    return spirals, track_fragments, boundary   

def parse_arguments() -> argparse.Namespace:
    #project_root = Path(__file__).parent.parent
    parser = argparse.ArgumentParser(
        description="CCL algorithm to find tracks of particles in images with noise."
    )
    parser.add_argument("--input","--inp", type=str, default="bubbles/resources/simulated.png")
    parser.add_argument("--gap_bridge","--gap", type=int, default=3)
    parser.add_argument("--noise-size","--noise","--n", type=int, default=5)
    parser.add_argument("--min_track_len","--mtl", type=int, default=10)
    parser.add_argument("--max-spur","--mxs","--ms", type=int, default=3)
    parser.add_argument("--elongation-ratio","--elr","--er", type=float, default=1.2)
    parser.add_argument("--simulate","--sim","--s", action="store_true")
    parser.add_argument("--bubble-chamber", "--bc", action="store_true")
    parser.add_argument("--min-skel-len", "--msl", type=int, default=15)
    parser.add_argument("--spiral-elong-max", "--sam", type=float, default=1.5)
    parser.add_argument("--len-range", "--lr", type=tuple, default=(30,500))
    parser.add_argument("--fragment-elong-min", "--fem", type=int, default=2)
    parser.add_argument("--boundary_frac", "--bf", type=float, default=0.05)
    return parser.parse_args()

def visualize_extraction(input_path,
                          gap_bridge=2,
                          noise_size=20,
                          min_skel_len=15,
                          max_spur=3,
                          spiral_elong_max=1.5,
                          spiral_len_range=(30, 500),
                          fragment_elong_min=2.0,
                          boundary_frac=0.05,
                          save_path="bebc_visualization.png"):
    """
    Full visualization of the CCL-based extraction pipeline for a BEBC image.

    Produces a 2x3 figure:
      Top row:    original → binary mask → labeled components
      Bottom row: spirals only → fragments only → combined overlay with annotations
    """
    img = Image.open(input_path).convert("L")
    gray = np.array(img, dtype=float)

    # ─── Pipeline ───
    gray_clean = median_filter(gray, size=3)
    fg = gray_clean < threshold_otsu(gray_clean)
    fg = remove_small_objects(fg, min_size=noise_size)
    fg = binary_closing(fg, disk(gap_bridge))

    labels = hoshen_kopelman(fg, connectivity=8)
    n_comp = labels.max()

    spirals, fragments, boundary_label = [], [], None

    for c in range(1, n_comp + 1):
        comp = (labels == c)
        area = comp.sum()

        if area > boundary_frac * fg.size:
            boundary_label = c
            continue

        skel = skeletonize(comp)
        skel = prune_spurs(skel, max_spur_length=max_spur)
        if skel.sum() < min_skel_len:
            continue

        paths = trace_skeleton_clean(skel)
        for p in paths:
            if len(p) < min_skel_len:
                continue
            rows, cols = p[:, 0], p[:, 1]
            h = rows.max() - rows.min() + 1
            w = cols.max() - cols.min() + 1
            elong = max(h, w) / max(min(h, w), 1)

            if elong < spiral_elong_max and spiral_len_range[0] < len(p) < spiral_len_range[1]:
                spirals.append((c, p))
            elif elong > fragment_elong_min:
                fragments.append((c, p))

    # ─── Figure ───
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("BEBC Bubble Chamber — CCL Track Extraction", fontsize=14, fontweight='bold')

    # Panel 1: Original
    axes[0, 0].imshow(img, cmap='gray')
    axes[0, 0].set_title("Original", fontsize=11)
    axes[0, 0].axis('off')

    # Panel 2: Binary mask
    axes[0, 1].imshow(fg, cmap='gray')
    axes[0, 1].set_title(f"Binary mask (Otsu + denoise, {fg.sum()} px)", fontsize=11)
    axes[0, 1].axis('off')

    # Panel 3: Labeled components (colored)
    if n_comp > 0:
        labeled_vis = np.zeros_like(labels, dtype=float)
        for c in range(1, n_comp + 1):
            labeled_vis[labels == c] = c
        im = axes[0, 2].imshow(labeled_vis, cmap='tab20', interpolation='nearest')
        axes[0, 2].set_title(f"Labeled components ({n_comp} total)", fontsize=11)
        axes[0, 2].axis('off')

    # Panel 4: Spirals only (zoomed to spiral region)
    axes[1, 0].imshow(img, cmap='gray')
    for c, p in spirals:
        axes[1, 0].plot(p[:, 1], p[:, 0], 'r-', lw=2.5, alpha=0.9)
        axes[1, 0].plot(p[0,1], p[0,0], 'o', color='yellow', ms=8, zorder=5,
                        path_effects=[pe.Stroke(linewidth=2, foreground='black')])
    # Zoom to spiral bounding box if spirals exist
    if spirals:
        all_spiral_pts = np.vstack([p for _, p in spirals])
        r_min = all_spiral_pts[:, 0].min() - 30
        r_max = all_spiral_pts[:, 0].max() + 30
        c_min = all_spiral_pts[:, 1].min() - 30
        c_max = all_spiral_pts[:, 1].max() + 30
        axes[1, 0].set_xlim(c_min, c_max)
        axes[1, 0].set_ylim(r_max, r_min)
    axes[1, 0].set_title(f"Spirals ({len(spirals)}) — zoomed", fontsize=11)
    axes[1, 0].axis('off')

    # Panel 5: Track fragments only
    axes[1, 1].imshow(img, cmap='gray')
    for i, (c, p) in enumerate(fragments):
        color = cm.tab20(i % 20 / 20)
        axes[1, 1].plot(p[:, 1], p[:, 0], '-', lw=1.5, color=color, alpha=0.7)
    axes[1, 1].set_title(f"Track fragments ({len(fragments)})", fontsize=11)
    axes[1, 1].axis('off')

    # Panel 6: Combined overlay with annotations
    axes[1, 2].imshow(img, cmap='gray')

    # Boundary
    if boundary_label is not None:
        b_mask = (labels == boundary_label)
        b_skel = skeletonize(b_mask)
        bys, bxs = np.where(b_skel)
        axes[1, 2].plot(bxs, bys, 'g-', lw=1.5, alpha=0.6, label='Boundary')

    # Spirals
    for c, p in spirals:
        axes[1, 2].plot(p[:, 1], p[:, 0], 'r-', lw=2.5, alpha=0.9)
        axes[1, 2].plot(p[0,1], p[0,0], 'o', color='yellow', ms=6, zorder=5)

    # Fragments
    for i, (c, p) in enumerate(fragments):
        color = cm.tab20(i % 20 / 20)
        axes[1, 2].plot(p[:, 1], p[:, 0], '-', lw=1, color=color, alpha=0.5)

    # Annotate spiral count
    if spirals:
        all_pts = np.vstack([p for _, p in spirals])
        cy = all_pts[:, 0].mean()
        cx = all_pts[:, 1].mean()
        axes[1, 2].annotate(f"{len(spirals)} spiral(s)\n(decay vertex)",
                            xy=(cx, cy), xytext=(cx + 80, cy - 80),
                            fontsize=10, color='red', fontweight='bold',
                            arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

    axes[1, 2].legend(loc='lower right', fontsize=9)
    axes[1, 2].set_title(f"Combined: {len(spirals)} spirals, {len(fragments)} fragments", fontsize=11)
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"  Spirals: {len(spirals)}")
    print(f"  Fragments: {len(fragments)}")
    print(f"  Boundary: {'detected' if boundary_label else 'not found'}")
    return spirals, fragments, boundary_label

# ──────────────────────────────────────────────
# 4. RUN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    arguments = parse_arguments()
    # input = Path("bubbles/resources/ABCMO_294_detail.png")
    input = Path(arguments.input)
    # path = make_noisy_bubble_chamber()
    if arguments.bubble_chamber:
        spirals, frags, boundary = visualize_extraction(input, arguments.gap_bridge, arguments.noise_size, arguments.min_skel_len, arguments.max_spur, arguments.spiral_elong_max, arguments.len_range, arguments.fragment_elong_min, arguments.boundary_frac)
    else:
        if not arguments.simulate:
            trajs = robust_extract_and_plot(input, arguments.gap_bridge, arguments.noise_size, arguments.min_track_len, arguments.max_spur, arguments.elongation_ratio)
        else:
            trajs = robust_extract_simulated(input, arguments.gap_bridge, arguments.noise_size, arguments.min_track_len, arguments.max_spur, arguments.elongation_ratio)
        print(f"\nFinal: {len(trajs)} tracks extracted")