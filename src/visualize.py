"""疊圖與標註輸出（FR-008、constitution III）。

一致配色：左眼藍、右眼紅；測量 A 連線綠；測量 B（兩動物右眼）連線橘。
於原圖疊加 mask、左右眼點、連線與距離數值。延遲載入 matplotlib。
"""

from __future__ import annotations

import os

import numpy as np

from src.models import AnimalInstance

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
