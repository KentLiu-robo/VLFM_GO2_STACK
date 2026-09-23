"""Frontier scoring using semantic map diffusion + geodesic distance.

Adapted from MSGNav/src/explore_utils.py (origin/CGFM branch).
All Habitat / TSDF references removed; works directly with obstacle_map from
vlfm_patch/obstacle_map.py.

Coordinate note
---------------
obstacle_map._xy_to_px(xy) returns [row, col] (index-0 = row, index-1 = col),
BUT _navigable_map / explored_area are filled as _map[col, row] — i.e. the
first numpy index is the column (x-pixel). This module converts to map-index
format [col, row] via _xy_to_map_idx before touching any map array.
"""

import heapq
import logging
import numpy as np
import scipy.ndimage


def _xy_to_map_idx(obstacle_map, xy_world: np.ndarray) -> np.ndarray:
    """Convert (N,2) world [x,y] → (N,2) map indices [mc, mr] for _navigable_map[mc, mr]."""
    px = obstacle_map._xy_to_px(np.atleast_2d(xy_world))  # returns (N,2) [row, col]
    return px[:, [1, 0]]  # swap → [col, row] = map-index convention


def _frontier_distances_bfs(
    navigable_mask: np.ndarray,
    start_idx: tuple,
    frontier_idxs: list,
    meters_per_cell: float,
) -> list:
    """Dijkstra from start_idx through navigable_mask to all frontier_idxs.

    All indices are (mc, mr) in map-index convention (mc = first numpy axis).

    Returns list of float distances in metres; falls back to Euclidean for
    unreachable frontiers.
    """
    h, w = navigable_mask.shape
    r0, c0 = int(round(start_idx[0])), int(round(start_idx[1]))

    target_keys: dict = {}
    for i, fp in enumerate(frontier_idxs):
        key = (int(round(fp[0])), int(round(fp[1])))
        target_keys.setdefault(key, []).append(i)

    euclidean = [
        float(np.linalg.norm([float(fp[0]) - r0, float(fp[1]) - c0])) * meters_per_cell
        for fp in frontier_idxs
    ]

    if not (0 <= r0 < h and 0 <= c0 < w and navigable_mask[r0, c0]):
        return euclidean

    dist_map = np.full((h, w), np.inf, dtype=np.float64)
    dist_map[r0, c0] = 0.0
    heap = [(0.0, r0, c0)]
    found: dict = {}

    _neighbors = [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, 1.4142136), (-1, 1, 1.4142136), (1, -1, 1.4142136), (1, 1, 1.4142136),
    ]

    while heap and len(found) < len(frontier_idxs):
        d, r, c = heapq.heappop(heap)
        if d > dist_map[r, c] + 1e-9:
            continue
        key = (r, c)
        if key in target_keys:
            for idx in target_keys[key]:
                if idx not in found:
                    found[idx] = d
        for dr, dc, cost in _neighbors:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and navigable_mask[nr, nc]:
                nd = d + cost
                if nd < dist_map[nr, nc]:
                    dist_map[nr, nc] = nd
                    heapq.heappush(heap, (nd, nr, nc))

    return [
        found[i] * meters_per_cell if i in found else euclidean[i]
        for i in range(len(frontier_idxs))
    ]


def select_frontier_by_score(
    obstacle_map,
    robot_xy: np.ndarray,
    semantic_map: np.ndarray,
    lambda_dist: float = 0.5,
    semantic_radius_m: float = 1.5,
    score_threshold: float = 0.05,
) -> int:
    """Return index into obstacle_map.frontiers of the best frontier.

    score(f) = mean_semantic_in_neighborhood(f) / (1 + lambda_dist * dist_norm(f))

    Falls back to nearest (argmin dist) when all scores <= score_threshold.

    Args:
        obstacle_map:       vlfm_patch ObstacleMap instance (provides frontiers,
                            _navigable_map, explored_area, pixels_per_meter,
                            _xy_to_px).
        robot_xy:           (2,) array [x, y] in world metres.
        semantic_map:       (H, W) float array, same shape as obstacle_map._navigable_map,
                            indexed [mc, mr] (map-index convention).
        lambda_dist:        Distance penalty weight.
        semantic_radius_m:  Neighbourhood radius for semantic aggregation.
        score_threshold:    Below this, fall back to distance-only.

    Returns:
        int index into obstacle_map.frontiers.
    """
    frontiers = obstacle_map.frontiers
    if len(frontiers) == 0:
        return 0
    if len(frontiers) == 1:
        return 0

    meters_per_cell = 1.0 / obstacle_map.pixels_per_meter

    start_idx = _xy_to_map_idx(obstacle_map, robot_xy.reshape(1, 2))[0]  # [mc, mr]
    frontier_idxs = _xy_to_map_idx(obstacle_map, frontiers)  # (N, 2) [mc, mr]

    traversable = obstacle_map._navigable_map & obstacle_map.explored_area
    shape = traversable.shape

    distances = _frontier_distances_bfs(
        navigable_mask=obstacle_map._navigable_map,
        start_idx=start_idx,
        frontier_idxs=frontier_idxs,
        meters_per_cell=meters_per_cell,
    )

    max_dist = max(distances) if distances else 1.0
    if max_dist < 1e-6:
        max_dist = 1.0

    R_cells = max(1, int(round(semantic_radius_m * obstacle_map.pixels_per_meter)))

    scores = []
    for i in range(len(frontiers)):
        mc, mr = int(round(frontier_idxs[i, 0])), int(round(frontier_idxs[i, 1]))
        seed = np.zeros(shape, dtype=bool)
        if 0 <= mc < shape[0] and 0 <= mr < shape[1]:
            seed[mc, mr] = True
        neighborhood = scipy.ndimage.binary_dilation(seed, mask=traversable, iterations=R_cells)
        if neighborhood.any():
            sem_score = float(semantic_map[neighborhood].mean())
        else:
            sem_score = 0.0

        dist_norm = distances[i] / max_dist
        score = sem_score / (1.0 + lambda_dist * dist_norm)
        scores.append(score)
        logging.info(
            f"[FrontierScore] f{i}: dist={distances[i]:.2f}m dist_norm={dist_norm:.3f} "
            f"sem={sem_score:.4f} score={score:.4f}"
        )

    max_score = max(scores)
    if max_score <= score_threshold:
        best = int(np.argmin(distances))
        logging.info(
            f"[FrontierScore] max_score={max_score:.4f} ≤ threshold={score_threshold}, "
            f"fallback to closest: frontier {best}"
        )
        return best

    best = int(np.argmax(scores))
    logging.info(f"[FrontierScore] best={best} score={max_score:.4f}")
    return best
