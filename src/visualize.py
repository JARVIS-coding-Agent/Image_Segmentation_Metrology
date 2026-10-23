"""疊圖與標註輸出（FR-008、constitution III）。

一致配色：左眼藍、右眼紅；測量 A 連線綠；測量 B（兩動物右眼）連線橘。
於原圖疊加 mask、左右眼點、連線與距離數值。延遲載入 matplotlib。
"""

from __future__ import annotations

import os

import numpy as np

from src.models import AnimalInstance
from src.segmentation import mask_to_original

LEFT_COLOR = "#1f77b4"   # 藍：左眼
RIGHT_COLOR = "#d62728"  # 紅：右眼
EYE_LINE_COLOR = "#2ca02c"   # 綠：測量 A 連線
PAIR_LINE_COLOR = "#ff7f0e"  # 橘：測量 B 連線


def render_overlay(
    image: np.ndarray,
    animals: list[AnimalInstance],
    output_path: str,
    right_eye_pairs: list[tuple[AnimalInstance, AnimalInstance, float]] | None = None,
) -> str:
    """繪製疊圖並存檔，回傳輸出路徑。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.axis("off")

    img_h, img_w = image.shape[0], image.shape[1]
    mask_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, a in enumerate(animals):
        # 畫 instance mask（半透明著色，P0-B）
        # 與 IoU/Dice 共用 mask_to_original，確保疊圖與評估用同一套對齊
        mask = mask_to_original(getattr(a, "mask", None), img_h, img_w)
        if mask is not None:
            color = mask_colors[idx % len(mask_colors)]
            rgb = tuple(int(color[i : i + 2], 16) / 255 for i in (1, 3, 5))
            overlay = np.zeros((img_h, img_w, 4), dtype=float)
            overlay[..., 0] = rgb[0]
            overlay[..., 1] = rgb[1]
            overlay[..., 2] = rgb[2]
            overlay[..., 3] = mask * 0.35  # alpha
            ax.imshow(overlay)

    for a in animals:
        le, re = a.left_eye, a.right_eye
        if le is not None:
            ax.plot(le.x, le.y, "o", color=LEFT_COLOR, markersize=8)
            ax.annotate("L", (le.x, le.y), color=LEFT_COLOR, fontsize=9)
        if re is not None:
            ax.plot(re.x, re.y, "o", color=RIGHT_COLOR, markersize=8)
            ax.annotate("R", (re.x, re.y), color=RIGHT_COLOR, fontsize=9)
        if a.measurable and le is not None and re is not None:
            ax.plot([le.x, re.x], [le.y, re.y], "-", color=EYE_LINE_COLOR, linewidth=2)
            import math

            d = math.hypot(le.x - re.x, le.y - re.y)
            mx, my = (le.x + re.x) / 2, (le.y + re.y) / 2
            ax.annotate(f"{d:.1f}px", (mx, my), color=EYE_LINE_COLOR, fontsize=9)

    for a, b, dist in right_eye_pairs or []:
        if a.right_eye is None or b.right_eye is None:
            continue
        ax.plot(
            [a.right_eye.x, b.right_eye.x],
            [a.right_eye.y, b.right_eye.y],
            "--",
            color=PAIR_LINE_COLOR,
            linewidth=2,
        )
        mx = (a.right_eye.x + b.right_eye.x) / 2
        my = (a.right_eye.y + b.right_eye.y) / 2
        ax.annotate(f"{dist:.1f}px", (mx, my), color=PAIR_LINE_COLOR, fontsize=9)

    fig.savefig(output_path, bbox_inches="tight", dpi=100)
    plt.close(fig)
    return output_path
