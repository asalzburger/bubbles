import numpy as np
from PIL import Image
from skimage.morphology import skeletonize, remove_small_objects
from collections import deque
import matplotlib.pyplot as plt
from pathlib import Path

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
    """
    Iteratively remove endpoint pixels `max_spur_length` times.
    This chops off all spurs shorter than max_spur_length pixels.
    """
    skel = skel.copy()
    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    for _ in range(max_spur_length):
        # Find all current endpoints (exactly 1 neighbor)
        to_remove = []
        for i in range(skel.shape[0]):
            for j in range(skel.shape[1]):
                if not skel[i, j]:
                    continue
                cnt = 0
                for di, dj in DIRS:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < skel.shape[0] and 0 <= nj < skel.shape[1] and skel[ni, nj]:
                        cnt += 1
                if cnt == 1:
                    to_remove.append((i, j))
        if not to_remove:
            break
        for i, j in to_remove:
            skel[i, j] = False

    return skel


def trace_skeleton_clean(skel):
    """
    Trace a pruned skeleton into trajectories, splitting at branch points
    by angle-continuity pairing.
    """
    ys, xs = np.where(skel)
    points = set(zip(ys.tolist(), xs.tolist()))
    if len(points) < 3:
        return [np.array(list(points), dtype=np.int32)] if points else []

    DIRS = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

    def neighbors(p):
        r, c = p
        return [(r+dr, c+dc) for dr, dc in DIRS if (r+dr, c+dc) in points]

    # Classify
    endpoints, branch_points = [], []
    for p in points:
        nd = len(neighbors(p))
        if nd == 1:
            endpoints.append(p)
        elif nd >= 3:
            branch_points.append(p)

    nodes = set(endpoints + branch_points)

    # Trace chains between nodes
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

    # Build unique edges
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

    # Direction of each edge as it leaves a given node
    def edge_dir(edge, at_node):
        a, b, path = edge
        if at_node == a:
            return (path[1][0]-path[0][0], path[1][1]-path[0][1])
        else:
            return (path[-2][0]-path[-1][0], path[-2][1]-path[-1][1])

    # Group edges by node
    from collections import defaultdict
    node_edges = defaultdict(list)
    for i, (a, b, path) in enumerate(edges):
        node_edges[a].append(i)
        node_edges[b].append(i)

    # Pair at branch points
    used = set()
    trajectories = []

    for bp in branch_points:
        eidxs = node_edges[bp]
        dirs = [edge_dir(edges[i], bp) for i in eidxs]

        # Pair by most-negative dot product (most opposite)
        paired = [False] * len(eidxs)
        for i in range(len(eidxs)):
            if paired[i]: continue
            best_j, best_dot = -1, 2.0
            for j in range(len(eidxs)):
                if j == i or paired[j]: continue
                dot = dirs[i][0]*dirs[j][0] + dirs[i][1]*dirs[j][1]
                if dot < best_dot:
                    best_dot = dot; best_j = j
            if best_j >= 0 and best_dot < 0:  # only pair if genuinely opposite
                paired[i] = paired[best_j] = True
                a_i, b_i, path_i = edges[eidxs[i]]
                a_j, b_j, path_j = edges[eidxs[best_j]]

                seg_i = path_i if bp == b_i else path_i[::-1]
                seg_j = path_j if bp == a_j else path_j[::-1]
                full = seg_i + seg_j[1:]
                trajectories.append(np.array(full, dtype=np.int32))
                used.add(eidxs[i]); used.add(eidxs[best_j])

        # Unpaired → dead-end segments
        for i in range(len(eidxs)):
            if not paired[i] and eidxs[i] not in used:
                a, b, path = edges[eidxs[i]]
                seg = path if bp == b else path[::-1]
                if len(seg) > 2:
                    trajectories.append(np.array(seg, dtype=np.int32))
                used.add(eidxs[i])

    # Safety: any unassigned edges
    for i, (a, b, path) in enumerate(edges):
        if i not in used:
            trajectories.append(np.array(path, dtype=np.int32))
            used.add(i)

    return [t for t in trajectories if len(t) > 3]


def extract_and_plot(input_path, threshold=0.3, min_size=30, max_spur=5):
    img = Image.open(input_path).convert("RGB")
    arr = np.array(img)
    mx = arr.max(axis=2).astype(float)
    mn = arr.min(axis=2).astype(float)
    sat = np.where(mx > 0, (mx - mn) / mx, 0)
    fg = sat > threshold

    labels = hoshen_kopelman(fg, connectivity=8)
    print(f"Connected components: {labels.max()}")

    all_trajs = []
    for c in range(1, labels.max() + 1):
        comp = (labels == c)
        if comp.sum() < min_size:
            print(f"  Component {c}: skipped (size {comp.sum()})")
            continue
        skel = skeletonize(comp)
        skel = prune_spurs(skel, max_spur_length=max_spur)
        paths = trace_skeleton_clean(skel)
        print(f"  Component {c}: {len(paths)} trajectory(ies)")
        for i, p in enumerate(paths):
            all_trajs.append((f"line_{c}_{i}", p))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    for name, path in all_trajs:
        ax.plot(path[:, 1], path[:, 0], linewidth=2.5, label=name, alpha=0.8)
        ax.plot(path[0,1], path[0,0], 'o', color='lime', markersize=10, zorder=5)
        ax.plot(path[-1,1], path[-1,0], 's', color='red', markersize=10, zorder=5)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig("trajectories_final.png", dpi=150, bbox_inches='tight')
    plt.show()
    return all_trajs

if __name__ == "__main__":
    input = Path("bubbles/resources/simulated.png")
    extract_and_plot(input, max_spur=5)   