"""Lightweight scene graph for CGFM navigation.

Stores detected objects as CLIP-embedded nodes with 3D world positions.
No SAM, no open3d — just YOLO-World bbox crops → CLIP embeddings.

Object merging: if a new detection's estimated world position is within
MERGE_DIST_M of an existing object, the two are merged using EMA on the
embedding and the position.

Usage
-----
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        'ViT-H-14', pretrained='laion2b_s32b_b79k')
    clip_model = clip_model.cuda().eval()
    sg = LightweightSceneGraph(clip_model, clip_preprocess, device='cuda')

    # inside perception loop:
    sg.update(color_rgb, detections, valid_idxs, depth_raw,
              depth_scale, tf_camera_to_episodic, fx, fy)

    # for frontier scoring:
    text_feat = sg.encode_text('chair')          # (1, D) tensor
    scored = sg.get_scored_objects(text_feat)    # [(score, world_xy), ...]
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

MERGE_DIST_M = 0.5   # merge if 3D world distance < this
EMA_ALPHA = 0.3      # new observation weight in EMA blend
CROP_PAD_PX = 20     # pixel padding around bbox before CLIP crop


class LightweightSceneGraph:
    """Grows an unordered set of scene-graph nodes, one per distinct 3D object."""

    def __init__(self, clip_model, clip_preprocess, device: str = "cuda"):
        self._clip = clip_model
        self._preprocess = clip_preprocess
        self._device = device
        self._objects: List[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all stored objects (call only on full episode reset)."""
        self._objects.clear()

    def update(
        self,
        color_rgb: np.ndarray,
        detections,
        valid_idxs: list,
        depth_raw: np.ndarray,
        depth_scale: float,
        tf_camera_to_episodic: np.ndarray,
        fx: float,
        fy: float,
    ) -> None:
        """Ingest one frame's worth of valid detections.

        Args:
            color_rgb:               (H, W, 3) uint8 RGB image.
            detections:              YOLO-World detection result with .boxes,
                                     .logits, .phrases (boxes are normalised
                                     [x1,y1,x2,y2] in [0,1]).
            valid_idxs:              Indices into detections that passed all
                                     prior gates.
            depth_raw:               (H, W) uint16 or float depth array.
            depth_scale:             Metres per raw depth unit.
            tf_camera_to_episodic:   (4,4) camera→world homogeneous transform.
            fx, fy:                  Focal lengths in pixels.
        """
        if not valid_idxs or detections is None:
            return

        H, W = color_rgb.shape[:2]

        for i in valid_idxs:
            box = detections.boxes[i].numpy() if hasattr(detections.boxes[i], "numpy") else np.asarray(detections.boxes[i])
            x1n, y1n, x2n, y2n = box

            # --- 3D world position ---
            world_xy = self._bbox_to_world_xy(
                box, depth_raw, depth_scale, fx, fy, W, H, tf_camera_to_episodic
            )
            if world_xy is None:
                continue

            # --- CLIP embedding of bbox crop ---
            clip_ft = self._crop_clip(color_rgb, x1n, y1n, x2n, y2n, W, H)
            if clip_ft is None:
                continue

            self._add_or_merge(clip_ft, world_xy)

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode a text query to a normalised CLIP feature (1, D)."""
        import open_clip
        tokenizer = open_clip.get_tokenizer("ViT-H-14")
        tokens = tokenizer([text]).to(self._device)
        with torch.no_grad():
            feat = self._clip.encode_text(tokens)
            feat = F.normalize(feat.float(), dim=-1)
        return feat

    def get_scored_objects(
        self, text_feat: torch.Tensor
    ) -> List[Tuple[float, np.ndarray]]:
        """Compute cosine similarity of every stored object to text_feat.

        Returns:
            List of (score, world_xy) sorted by score descending.
            score is raw cosine similarity in [-1, 1].
        """
        if not self._objects:
            return []
        results = []
        for obj in self._objects:
            ft = obj["clip_ft"].to(self._device)  # (1, D)
            score = float((ft @ text_feat.T).squeeze())
            results.append((score, obj["world_xy"].copy()))
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    def __len__(self) -> int:
        return len(self._objects)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _bbox_to_world_xy(
        self,
        box: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale: float,
        fx: float,
        fy: float,
        W: int,
        H: int,
        tf_camera_to_episodic: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Back-project box-center pixel to world (x, y) via depth.

        Returns (2,) array or None if no valid depth at that pixel.
        Mirrors estimate_target_xy() in run_vlfm_pipeline.py.
        """
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2 * W)
        cy = int((y1 + y2) / 2 * H)

        PAD = 3
        r0 = max(0, cy - PAD); r1 = min(H, cy + PAD + 1)
        c0 = max(0, cx - PAD); c1 = min(W, cx + PAD + 1)
        patch = depth_raw[r0:r1, c0:c1]
        valid = patch[patch > 0]
        if len(valid) == 0:
            return None
        depth_m = float(np.median(valid)) * depth_scale
        if depth_m < 0.1 or depth_m > 10.0:
            return None

        x_cam = (cx - W / 2) * depth_m / fx
        y_cam = (cy - H / 2) * depth_m / fy
        # Camera convention matches get_point_cloud: (forward, left, up)
        point_cam = np.array([depth_m, -x_cam, -y_cam, 1.0])
        point_world = tf_camera_to_episodic @ point_cam
        return point_world[:2].copy()

    def _crop_clip(
        self,
        color_rgb: np.ndarray,
        x1n: float,
        y1n: float,
        x2n: float,
        y2n: float,
        W: int,
        H: int,
    ) -> Optional[torch.Tensor]:
        """Crop bbox with padding, run through CLIP, return normalised (1,D) tensor."""
        x1 = max(0, int(x1n * W) - CROP_PAD_PX)
        y1 = max(0, int(y1n * H) - CROP_PAD_PX)
        x2 = min(W, int(x2n * W) + CROP_PAD_PX)
        y2 = min(H, int(y2n * H) + CROP_PAD_PX)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = color_rgb[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop)
        img_tensor = self._preprocess(pil_img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            feat = self._clip.encode_image(img_tensor)
            feat = F.normalize(feat.float(), dim=-1)
        return feat.cpu()

    def _add_or_merge(self, clip_ft: torch.Tensor, world_xy: np.ndarray) -> None:
        """Add new object or EMA-merge with nearest existing one."""
        best_idx = -1
        best_dist = MERGE_DIST_M

        for i, obj in enumerate(self._objects):
            d = float(np.linalg.norm(world_xy - obj["world_xy"]))
            if d < best_dist:
                best_dist = d
                best_idx = i

        if best_idx >= 0:
            obj = self._objects[best_idx]
            obj["clip_ft"] = F.normalize(
                EMA_ALPHA * clip_ft + (1 - EMA_ALPHA) * obj["clip_ft"], dim=-1
            )
            obj["world_xy"] = (
                EMA_ALPHA * world_xy + (1 - EMA_ALPHA) * obj["world_xy"]
            )
            obj["count"] += 1
        else:
            self._objects.append({
                "clip_ft": clip_ft,
                "world_xy": world_xy.copy(),
                "count": 1,
            })
            logging.info(
                f"[SceneGraph] added object #{len(self._objects)}: "
                f"xy=({world_xy[0]:.2f},{world_xy[1]:.2f})"
            )
