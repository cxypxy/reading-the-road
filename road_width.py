"""
Reading the Road - Road Width Estimator

Core pipeline:
image -> road mask -> boundaries -> geometry -> width + confidence
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class WidthResult:
    width_m: float
    std_m: float
    confidence: float
    trust: str
    horizon_v: int
    vanishing_u: int
    left_line: tuple
    right_line: tuple
    mask: np.ndarray
    overlay: np.ndarray
    per_row_widths: list
    warnings: list[str] = field(default_factory=list)


def preprocess(
    image: np.ndarray,
    target_w: int = 1280,
    target_h: int = 720,
) -> np.ndarray:
    """Resize the image and improve local contrast."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    img = cv2.resize(image, (target_w, target_h))
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    enhanced = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def segment_road_grabcut(img: np.ndarray) -> np.ndarray:
    """Segment the road using GrabCut, with a conservative fallback."""
    h, w = img.shape[:2]

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    rect = (
        int(w * 0.10),
        int(h * 0.40),
        int(w * 0.80),
        int(h * 0.58),
    )

    try:
        cv2.grabCut(
            img,
            mask,
            rect,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return fallback_road_mask(h, w)

    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    binary = keep_bottom_component(binary)
    binary = largest_component(binary)

    if cv2.countNonZero(binary) < int(h * w * 0.01):
        return fallback_road_mask(h, w)

    return binary


def fallback_road_mask(h: int, w: int) -> np.ndarray:
    """Create a conservative trapezoid-shaped road mask."""
    mask = np.zeros((h, w), dtype=np.uint8)

    polygon = np.array(
        [
            [int(w * 0.42), int(h * 0.42)],
            [int(w * 0.58), int(h * 0.42)],
            [w - 1, h - 1],
            [0, h - 1],
        ],
        dtype=np.int32,
    )

    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def largest_component(binary: np.ndarray) -> np.ndarray:
    """Keep only the largest connected foreground component."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return binary

    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def keep_bottom_component(binary: np.ndarray) -> np.ndarray:
    """Keep foreground components touching the bottom image edge."""
    h, _ = binary.shape
    if h == 0:
        return binary

    num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if num_labels <= 1:
        return binary

    bottom_row = labels[h - 1, :]
    valid_labels = set(np.unique(bottom_row[bottom_row > 0]))
    if not valid_labels:
        return binary

    kept = np.zeros_like(binary)
    for label in valid_labels:
        kept[labels == label] = 255

    return kept.astype(np.uint8)


def _estimate_boundaries(mask: np.ndarray):
    """Return the left and right boundary points for lane-like road edges."""
    h, w = mask.shape[:2]
    left_points = []
    right_points = []
    start_y = max(0, h // 2)

    for y in range(h - 1, start_y - 1, -1):
        ys = np.where(mask[y] > 0)[0]
        if ys.size == 0:
            continue
        left = int(ys.min())
        right = int(ys.max())
        if right - left < max(5, w * 0.02):
            continue
        left_points.append((left, y))
        right_points.append((right, y))

    if not left_points or not right_points:
        lane_x = w * 0.5
        left_points = [(int(lane_x * 0.35), h - 1), (int(lane_x * 0.4), h // 2)]
        right_points = [(int(lane_x * 1.65), h - 1), (int(lane_x * 1.6), h // 2)]

    return left_points, right_points


def _fit_line(points):
    """Fit a straight line to points of the form (x, y)."""
    if len(points) < 2:
        return 0.0, 0.0
    xs = np.array([x for x, _ in points], dtype=np.float32)
    ys = np.array([y for _, y in points], dtype=np.float32)
    slope, intercept = np.polyfit(ys, xs, 1)
    return float(slope), float(intercept)


def _intersection(m1: float, b1: float, m2: float, b2: float):
    """Compute road-edge intersection as a vanishing point."""
    if abs(m1 - m2) < 1e-6:
        return 0, 0
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return float(x), float(y)


def _row_widths(mask: np.ndarray):
    """Collect each row's visible road width in pixels."""
    h, _ = mask.shape
    widths = []
    for y in range(h - 1, max(0, h // 2) - 1, -1):
        pixels = np.where(mask[y] > 0)[0]
        if pixels.size < 2:
            continue
        left = int(pixels.min())
        right = int(pixels.max())
        widths.append(float(right - left))
    if not widths:
        widths = [float(mask.shape[1] * 0.25)]
    return widths


def measure_road(image: np.ndarray, camera_height_m: float = 1.5) -> WidthResult:
    """Estimate road width in metres from a single road image."""
    processed = preprocess(image)
    mask = segment_road_grabcut(processed)

    h, w = mask.shape[:2]
    left_points, right_points = _estimate_boundaries(mask)
    left_slope, left_intercept = _fit_line(left_points)
    right_slope, right_intercept = _fit_line(right_points)
    vanishing_u, vanishing_v = _intersection(left_slope, left_intercept, right_slope, right_intercept)

    if vanishing_u == 0 and vanishing_v == 0:
        vanishing_u = w * 0.5
        vanishing_v = h * 0.5

    vanishing_u = int(np.clip(vanishing_u, 0, w - 1))
    vanishing_v = int(np.clip(vanishing_v, 0, h - 1))

    widths = _row_widths(mask)
    median_width_px = float(np.median(widths))
    std_px = float(np.std(widths))

    horizon_v = int(np.clip(vanishing_v, int(h * 0.15), h - 1))
    perspective_factor = max(1.0, (h - horizon_v) / 3.0)
    width_m = max(
        1.0,
        (median_width_px / max(1.0, w * 0.55))
        * (14.0 + camera_height_m * 5.0)
        / (perspective_factor / 30.0),
    )
    std_m = max(
        0.3,
        (std_px / max(1.0, w * 0.55))
        * (14.0 + camera_height_m * 5.0)
        / (perspective_factor / 30.0),
    )

    mask_ratio = cv2.countNonZero(mask) / float(mask.size)
    consistency = 1.0 - min(1.0, std_px / max(1.0, median_width_px))
    coverage = min(1.0, mask_ratio / 0.12)
    confidence = float(np.clip(0.45 * coverage + 0.40 * consistency + 0.15, 0.0, 1.0))

    warnings = []
    if abs(camera_height_m - 1.5) < 0.1:
        warnings.append("Camera height was assumed to be 1.5 m because it was not provided.")
    if confidence < 0.45:
        warnings.append("Low confidence: try a clearer photo with a more visible road surface.")
    if vanishing_v > h * 0.8:
        warnings.append("The detected vanishing point is unusually low in the frame.")

    if confidence >= 0.75:
        trust = "High"
    elif confidence >= 0.45:
        trust = "Medium"
    else:
        trust = "Low"

    overlay = processed.copy()
    overlay[mask > 0] = (0, 255, 0)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    left_x = int(np.clip(left_slope * vanishing_v + left_intercept, 0, w - 1))
    right_x = int(np.clip(right_slope * vanishing_v + right_intercept, 0, w - 1))

    cv2.line(overlay, (left_x, vanishing_v), (left_x, h - 1), (255, 0, 0), 2)
    cv2.line(overlay, (right_x, vanishing_v), (right_x, h - 1), (255, 0, 0), 2)
    cv2.circle(overlay, (vanishing_u, vanishing_v), 6, (0, 0, 255), -1)

    return WidthResult(
        width_m=float(width_m),
        std_m=float(std_m),
        confidence=float(confidence),
        trust=trust,
        horizon_v=horizon_v,
        vanishing_u=vanishing_u,
        left_line=(left_slope, left_intercept),
        right_line=(right_slope, right_intercept),
        mask=mask,
        overlay=overlay,
        per_row_widths=widths,
        warnings=warnings,
    )


__all__ = ["WidthResult", "preprocess", "segment_road_grabcut", "measure_road"]
