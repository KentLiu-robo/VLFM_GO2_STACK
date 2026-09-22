# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import threading
from typing import List, Optional

import numpy as np

from vlfm.vlm.detections import ObjectDetections

from .server_wrapper import ServerMixin, host_model, send_request, str_to_image

try:
    from ultralytics import YOLO
except ModuleNotFoundError:
    print("Could not import ultralytics. This is OK if you are only using the client.")

YOLO_WORLD_WEIGHTS = "data/yolov8s-worldv2.pt"
CLASSES = "chair . person . dog ."  # Default classes, same caption format as GroundingDINO


def _caption_to_classes(caption: str) -> List[str]:
    """Splits a GroundingDINO-style ' . '-separated caption into a plain
    class-name list, e.g. 'fan . fans . ceiling fan .' -> ['fan', 'fans',
    'ceiling fan']."""
    caption = caption.strip()
    if caption.endswith(" ."):
        caption = caption[: -len(" .")]
    return [c.strip() for c in caption.split(" . ") if c.strip()]


class YOLOWorld:
    """Open-vocabulary detector (YOLO-World via ultralytics) -- same
    ' . '-separated caption convention as GroundingDINO, but YOLO-World
    itself only supports one flat, mutually-exclusive class list at a time
    (no exact-phrase re-filtering needed server-side like GroundingDINO
    does), so every returned box is already about a class the caller asked
    for."""

    def __init__(self, weights: str = YOLO_WORLD_WEIGHTS, caption: str = CLASSES):
        self.model = YOLO(weights)
        self._classes: Optional[List[str]] = None
        # Flask serves each request on its own thread, and the client's 1s
        # read timeout is shorter than the first set_classes() call (CLIP text
        # encode, ~3s) -- so the client retried while the first request was
        # still running, and several threads mutated the same model at once
        # (reproduced 2026-09-18: afterwards the server ignored the requested
        # classes and returned the default 80 COCO labels for every request).
        # One lock serializes set_classes+predict so retries just wait.
        self._lock = threading.Lock()
        self.set_caption(caption)
        # Warm-up: builds the ultralytics predictor / CUDA kernels up front so
        # the first real request isn't slow.
        self.model.predict(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)

    def set_caption(self, caption: str) -> None:
        classes = _caption_to_classes(caption)
        if classes != self._classes:
            self.model.set_classes(classes)
            self._classes = classes

    def predict(
        self, image: np.ndarray, caption: Optional[str] = None, conf_thres: float = 0.05
    ) -> ObjectDetections:
        """
        Arguments:
            image (np.ndarray): An RGB image as a numpy array.
            caption (Optional[str]): ' . '-separated class names, same format
                as GroundingDINO's caption. If not provided, the classes set
                at construction time (or the last call to set_caption) are
                reused.
            conf_thres (float): Confidence threshold -- kept low here since
                the caller (run_vlfm_pipeline.py) does its own thresholding
                on top of this; this only avoids wasting time on near-zero
                detections.
        """
        with self._lock:
            if caption is not None:
                self.set_caption(caption)
            # ultralytics treats numpy input as BGR (OpenCV convention), but
            # callers here pass RGB -- feeding RGB unflipped swaps red/blue and
            # measurably lowers confidence (live trash-can frame: 0.21-0.28 ->
            # 0.34-0.46 once flipped; tv ≥0.25 frames 8 -> 24 offline).
            bgr = np.ascontiguousarray(image[..., ::-1])
            results = self.model.predict(bgr, conf=conf_thres, verbose=False)[0]
            boxes = results.boxes.xyxyn.cpu()
            logits = results.boxes.conf.cpu()
            phrases = [results.names[int(c)] for c in results.boxes.cls]

        return ObjectDetections(boxes, logits, phrases, image_source=image, fmt="xyxy")


class YOLOWorldClient:
    def __init__(self, port: int = 12185):
        self.url = f"http://localhost:{port}/yolo_world"

    def predict(self, image_numpy: np.ndarray, caption: Optional[str] = "") -> ObjectDetections:
        response = send_request(self.url, image=image_numpy, caption=caption)
        detections = ObjectDetections.from_json(response, image_source=image_numpy)

        return detections


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=12185)
    args = parser.parse_args()

    print("Loading model...")

    class YOLOWorldServer(ServerMixin, YOLOWorld):
        def process_payload(self, payload: dict) -> dict:
            image = str_to_image(payload["image"])
            caption = payload.get("caption") or None
            return self.predict(image, caption=caption).to_json()

    yolo_world = YOLOWorldServer()
    print("Model loaded!")
    print(f"Hosting on port {args.port}...")
    host_model(yolo_world, name="yolo_world", port=args.port)
