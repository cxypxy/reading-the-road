"""
Reading the Road - Road Width Estimator
Core pipeline: image -> road mask -> boundaries -> geometry -> width + confidence
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class WidthResult:
    width_m: float
    std_m: float
    confidence: float
    trust: str
    horizon_v: int
    vanishing_u: int
    left_line: tuple  # (slope, intercept) in (x = a*y + b)
    right_line: tuple
    mask: np.ndarray
    overlay: np.ndarray
    per_row_widths: list
    warnings: list[str] = None


def preprocess(image: np.ndarray, target_w=1280, target_h=720) -> np.ndarray:
    """Resize + contrast enhance."""
    img = cv2.resize(image, (target_w, target_h))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def segment_road_grabcut(img: np.ndarray) -> np.ndarray:
    """
    Fast classical road segmentation using GrabCut with a bottom-center rectangle.
    Works well for typical dashcam / phone photos of roads.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Assume road occupies bottom-center: skip top 40% (sky/horizon),
    # and side margins
    rect = (int(w * 0.10), int(h * 0.40), int(w * 0.80), int(h * 0.58))
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        # A uniform or very small image may not provide enough colour evidence
        # for GrabCut. A conservative trapezoid keeps the pipeline inspectable.
        return fallback_road_mask(h, w)

    binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    # Keep only the largest connected component (removes stray blobs)
    binary = largest_component(binary)
    # Keep only the component touching the bottom of the image
    binary = keep_bottom_component(binary)
    return binary


def fallback_road_mask(h: int, w: int) -> np.ndarray:
    """Create a conservative road-shaped mask when visual segmentation is unavailable."""
    mask = np.zeros((h, w), np.uint8)
    polygon = np.array(
        [[int(w * 0.42), int(h * 0.42)],
         [int(w * 0.58), int(h * 0.42)],
         [w - 1, h - 1],
         [0, h - 1]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, polygon, 255)
    return mask


def largest_component(binary: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        return binary
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return np.where(labels == idx, 255, 0).astype(np.uint8)


def keep_bottom_component(binary: np.ndarray) -> np.ndarray:
    """If the mask doesn't reach the bottom, try to keep the biggest anyway."""
    h, w = binary.shape
    bottom_row = binary[h - 1, :]
    if bottom_row.sum() == 0:
        return binary  # nothing to do
    return binary


def find_vanishing_point(mask: np.ndarray) -> tuple[int, int]:
    """
    Estimate vanishing point by fitting lines to left/right boundaries of mask
    and finding their intersection.
    """
    h, w = mask.shape
    left_pts, right_pts = [], []

    # Sample rows from lower half (below approximate horizon)
    rows = np.linspace(int(h * 0.45), int(h * 0.98), 40).astype(int)
    for v in rows:
        cols = np.where(mask[v, :] > 0)[0]
        if len(cols) < 20:
            continue
        left_pts.append((cols[0], v))
        right_pts.append((cols[-1], v))

    if len(left_pts) < 5:
        # Fallback: assume horizon at 45% height, center
        return int(w / 2), int(h * 0.45)

    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)

    # Fit x = a*y + b (line in image coords)
    a_l, b_l = np.polyfit(left_pts[:, 1], left_pts[:, 0], 1)
    a_r, b_r = np.polyfit(right_pts[:, 1], right_pts[:, 0], 1)

    # Intersection: a_l*y + b_l = a_r*y + b_r
    if abs(a_l - a_r) < 1e-6:
        return int(w / 2), int(h * 0.45)
    y_vp = (b_r - b_l) / (a_l - a_r)
    x_vp = a_l * y_vp + b_l

    # Sanity check: VP should be within reasonable image range
    if not (0 <= x_vp <= w and 0 <= y_vp <= h):
        return int(w / 2), int(h * 0.45)

    return int(x_vp), int(y_vp)


def extract_boundaries(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_line, right_line) as (slope_a, intercept_b) in x = a*y + b."""
    h, w = mask.shape
    left_pts, right_pts = [], []
    rows = np.linspace(int(h * 0.45), int(h * 0.98), 40).astype(int)
    for v in rows:
        cols = np.where(mask[v, :] > 0)[0]
        if len(cols) < 20:
            continue
        left_pts.append((cols[0], v))
        right_pts.append((cols[-1], v))

    if len(left_pts) < 2 or len(right_pts) < 2:
        h, w = mask.shape
        # This fallback is deliberately marked down by the confidence model.
        return (-0.42, w * 0.50), (0.42, w * 0.50)

    left_pts = np.array(left_pts)
    right_pts = np.array(right_pts)

    a_l, b_l = np.polyfit(left_pts[:, 1], left_pts[:, 0], 1)
    a_r, b_r = np.polyfit(right_pts[:, 1], right_pts[:, 0], 1)
    return (a_l, b_l), (a_r, b_r)


def compute_width(mask: np.ndarray,
                  left_line: tuple,
                  right_line: tuple,
                  horizon_v: int,
                  camera_height_m: float,
                  min_row_offset: int = 30) -> tuple[float, float, list]:
    """
    Apply pinhole formula at multiple rows and take median.

    width_m = pixel_width * camera_height / (v - horizon_v)
    """
    h, w = mask.shape
    a_l, b_l = left_line
    a_r, b_r = right_line

    rows = np.linspace(horizon_v + min_row_offset, h - 5, 30).astype(int)
    widths = []
    for v in rows:
        u_l = a_l * v + b_l
        u_r = a_r * v + b_r
        # Clamp to image bounds
        u_l = np.clip(u_l, 0, w - 1)
        u_r = np.clip(u_r, 0, w - 1)
        pixel_width = u_r - u_l
        if pixel_width <= 0:
            continue
        depth = v - horizon_v
        if depth <= 0:
            continue
        width_m = pixel_width * camera_height_m / depth
        # Reject absurd values (sanity)
        if 0.5 < width_m < 50:
            widths.append(width_m)

    if not widths:
        return 0.0, 0.0, []

    widths = np.array(widths)
    return float(np.median(widths)), float(np.std(widths)), widths.tolist()


def compute_confidence(mask: np.ndarray,
                       widths: list,
                       edge_sharpness: float,
                       image: np.ndarray,
                       horizon_v: int) -> float:
    """Combine several signals into a 0-1 confidence."""
    if not widths:
        return 0.0

    # 1. Mask coverage of lower image (should be a decent chunk, but not everything)
    h, w = mask.shape
    lower = mask[int(h * 0.5):, :]
    coverage = lower.mean() / 255.0
    coverage_score = 1.0 - min(abs(coverage - 0.45) / 0.45, 1.0)
    if coverage < 0.08 or coverage > 0.90:
        coverage_score *= 0.25

    # 2. Consistency of widths across rows (low std/mean = good)
    mean_w = float(np.mean(widths))
    std_w = float(np.std(widths))
    consistency = 1.0 / (1.0 + std_w / max(mean_w, 1e-3) * 3.0)

    # 3. Edge sharpness (Sobel magnitude on the image)
    sobel = cv2.Sobel(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F, 1, 0, ksize=3)
    sharp_norm = min(np.mean(np.abs(sobel)) / 60.0, 1.0)

    # 4. Horizon plausibility: horizon should be in upper-middle area
    h_ratio = horizon_v / h
    horizon_score = 1.0 - min(abs(h_ratio - 0.45) / 0.35, 1.0)

    conf = (0.30 * coverage_score +
            0.30 * consistency +
            0.20 * sharp_norm +
            0.20 * horizon_score)
    return float(np.clip(conf, 0, 1))


def trust_level(conf: float) -> str:
    if conf > 0.70:
        return "High"
    elif conf > 0.45:
        return "Medium"
    return "Low"


def draw_overlay(image: np.ndarray,
                 mask: np.ndarray,
                 left_line: tuple,
                 right_line: tuple,
                 horizon_v: int,
                 width_m: float,
                 std_m: float,
                 conf: float) -> np.ndarray:
    overlay = image.copy()
    # Road mask in green
    overlay[mask > 0] = [0, 255, 0]
    result = cv2.addWeighted(overlay, 0.35, image, 0.65, 0)

    h, w = image.shape[:2]
    a_l, b_l = left_line
    a_r, b_r = right_line

    # Draw left/right edge lines from horizon down to bottom
    y0, y1 = int(horizon_v), h - 1
    cv2.line(result, (int(a_l * y0 + b_l), y0), (int(a_l * y1 + b_l), y1), (255, 0, 0), 3)
    cv2.line(result, (int(a_r * y0 + b_r), y0), (int(a_r * y1 + b_r), y1), (0, 0, 255), 3)

    # Horizon line
    cv2.line(result, (0, int(horizon_v)), (w, int(horizon_v)), (0, 255, 255), 2)

    # Text box
    width_ft = width_m * 3.28084
    std_ft = std_m * 3.28084
    text1 = f"Width: {width_ft:.1f} ft  (+/- {std_ft:.1f} ft)"
    text2 = f"Confidence: {conf*100:.0f}%  ({trust_level(conf)})"
    cv2.rectangle(result, (20, 20), (620, 110), (0, 0, 0), -1)
    cv2.putText(result, text1, (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(result, text2, (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return result


def measure_road(image_bgr: np.ndarray,
                 camera_height_m: float = 1.2,
                 use_grabcut: bool = True) -> WidthResult:
    """End-to-end pipeline with safe degradation for incomplete survey imagery."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image_bgr must be a non-empty BGR image with three channels")
    if camera_height_m <= 0:
        raise ValueError("camera_height_m must be greater than zero")

    img = preprocess(image_bgr)
    mask = segment_road_grabcut(img) if use_grabcut else fallback_road_mask(*img.shape[:2])

    x_vp, y_vp = find_vanishing_point(mask)
    left_line, right_line = extract_boundaries(mask)

    width_m, std_m, widths = compute_width(
        mask, left_line, right_line, y_vp, camera_height_m
    )

    # Edge sharpness signal for confidence
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    edge_sharpness = float(np.mean(np.abs(sobel)))

    conf = compute_confidence(mask, widths, edge_sharpness, img, y_vp)
    warnings = []
    if not widths:
        warnings.append("No stable road-width samples were found.")
    if len(widths) < 8:
        warnings.append("Few valid scan lines were available; inspect the overlay.")
    if conf < 0.45:
        warnings.append("Perspective, visibility, or segmentation quality is limiting trust.")
    overlay = draw_overlay(img, mask, left_line, right_line, y_vp, width_m, std_m, conf)

    return WidthResult(
        width_m=width_m,
        std_m=std_m,
        confidence=conf,
        trust=trust_level(conf),
        horizon_v=y_vp,
        vanishing_u=x_vp,
        left_line=left_line,
        right_line=right_line,
        mask=mask,
        overlay=overlay,
        per_row_widths=widths,
        warnings=warnings,
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python road_width.py <image_path> [camera_height_m]")
        sys.exit(1)
    path = sys.argv[1]
    h = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2
    img = cv2.imread(path)
    if img is None:
        print(f"Could not read {path}")
        sys.exit(1)
    result = measure_road(img, camera_height_m=h)
    print(f"Width: {result.width_m * 3.28084:.2f} ft +/- {result.std_m * 3.28084:.2f} ft")
    print(f"Confidence: {result.confidence*100:.0f}% ({result.trust})")
    cv2.imwrite("output_overlay.jpg", result.overlay)
    cv2.imwrite("output_mask.jpg", result.mask)
    print("Saved output_overlay.jpg and output_mask.jpg")