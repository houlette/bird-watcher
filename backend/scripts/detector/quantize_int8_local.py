"""INT8 post-training quantization of the dynamic-shape yolo11s OpenVINO FP32 IR, run on the Mac.

Tiles are letterboxed as Ultralytics does for PyTorch: kept at their own size and
padded only to a multiple of 32, so calibration sees the shapes serving will.

Calibration: 3 tiles from each of 40 unlabelled saved source frames (listed in
calibration_frames.txt), letterboxed to 1024 square exactly as Ultralytics does
for a static model. The detection head's arithmetic, DFL and sigmoid stay in
floating point, as in Ultralytics' own INT8 export. Only activation ranges are
computed here; the result is benchmarked and quality-checked on the VM.
"""
import random
import shutil
from pathlib import Path

import cv2
import nncf
import numpy as np
import openvino as ov

HERE = Path(__file__).parent
TILE_PX = 1024
OVERLAP = int(TILE_PX * 0.20)
random.seed(0)


def tile_offsets(width, height):
    # Same as pipeline.detect._tile_offsets.
    step = TILE_PX - OVERLAP
    rects, y = [], 0
    while y < height:
        x = 0
        while x < width:
            rects.append((x, y, min(TILE_PX, width - x), min(TILE_PX, height - y)))
            if x + TILE_PX >= width:
                break
            x += step
        if y + TILE_PX >= height:
            break
        y += step
    return rects


def letterbox_square(tile):
    h, w = tile.shape[:2]
    r = min(TILE_PX / h, TILE_PX / w)
    nh, nw = round(h * r), round(w * r)
    if (nh, nw) != (h, w):
        tile = cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, left = (TILE_PX - nh) // 2, (TILE_PX - nw) // 2
    out = np.full((TILE_PX, TILE_PX, 3), 114, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = tile
    return out


def letterbox_rect(tile):
    h, w = tile.shape[:2]
    ph, pw = (-h) % 32, (-w) % 32
    top, left = ph // 2, pw // 2
    return cv2.copyMakeBorder(tile, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))


tiles = []
for name in (HERE / "calibration_frames.txt").read_text().split():
    img = cv2.imread(str(HERE / "frames" / name))
    for x, y, tw, th in random.sample(tile_offsets(img.shape[1], img.shape[0]), 3):
        lb = letterbox_rect(img[y:y + th, x:x + tw])
        tiles.append(np.ascontiguousarray(lb[..., ::-1].transpose(2, 0, 1))[None])
print(f"calibration tiles: {len(tiles)}")

src = HERE / "yolo11s_ov_dynamic_openvino_model"
model = ov.Core().read_model(src / "yolo11s_ov_dynamic.xml")
ignored = nncf.IgnoredScope(
    patterns=[".*model.23/.*/Add", ".*model.23/.*/Sub*", ".*model.23/.*/Mul*", ".*model.23/.*/Div*", ".*model.23\\.dfl.*"],
    types=["Sigmoid"],
    validate=False,
)
int8 = nncf.quantize(
    model,
    nncf.Dataset(tiles, lambda t: t.astype(np.float32) / 255.0),
    preset=nncf.QuantizationPreset.MIXED,
    subset_size=len(tiles),
    ignored_scope=ignored,
)
out = HERE / "yolo11s_ov_int8dyn_openvino_model"
out.mkdir(exist_ok=True)
ov.save_model(int8, out / "yolo11s_ov_int8dyn.xml")
shutil.copy(src / "metadata.yaml", out / "metadata.yaml")
print(f"saved {out}")
