import cv2
import numpy as np


def extract_clothing_region(frame, person_bbox, face_bbox=None, margin_erosion=0.08):
    """
    Extracts a tightly-bounded upper-torso (chest) region, invariant to raised arms,
    body-pose expansion, and loose detector bounding boxes.

    Key design choices vs. a naive face-relative crop:
      - Crop width is anchored to face width but the *vertical* band is kept short and
        pinned close to the collarbone line, well below shoulder/arm height, so raised
        or extended arms (which move roughly horizontally at shoulder height) fall
        outside the box entirely rather than needing to be filtered out after the fact.
      - An inward erosion margin is applied after clamping, shrinking the box a fixed
        percentage on all sides. This guards against detector bbox jitter/looseness
        bleeding background or clothing-edge (shadow/highlight) pixels into the sample.

    Args:
        frame (np.ndarray): Full video/image frame in BGR format.
        person_bbox (list or tuple): [x1, y1, x2, y2] person bounding box.
        face_bbox (list or tuple, optional): [fx1, fy1, fx2, fy2] face bounding box.
        margin_erosion (float): Fraction of crop width/height to shave off each edge
            after the initial box is computed. 0.08 = 8% inward on every side.

    Returns:
        np.ndarray: Cropped BGR region of the torso. Empty array if no valid region.
    """
    fh, fw = frame.shape[:2]
    px1, py1, px2, py2 = map(int, person_bbox)
    px1, px2 = max(0, min(px1, fw)), max(0, min(px2, fw))
    py1, py2 = max(0, min(py1, fh)), max(0, min(py2, fh))
    w, h = px2 - px1, py2 - py1

    if face_bbox is not None:
        fx1, fy1, fx2, fy2 = map(int, face_bbox)
        face_w = max(1, fx2 - fx1)
        face_h = max(1, fy2 - fy1)
        face_cx = (fx1 + fx2) // 2

        # Narrower than before (0.55 vs 0.7) — the widest a torso appears just below
        # the collar, before shoulders flare out, tracks closer to face width itself.
        cx1 = int(face_cx - 0.55 * face_w)
        cx2 = int(face_cx + 0.55 * face_w)
        # Start right at the chin and stop before the shoulder line typically begins
        # to widen (~1.6x face height below chin is a safe pre-shoulder cutoff for
        # most adult proportions), instead of extending to 2.5x.
        cy1 = int(fy2 + 0.35 * face_h)
        cy2 = int(fy2 + 1.6 * face_h)
    else:
        # Fallback: narrow vertical strip centered in the person box, avoiding the
        # outer 40% on each side where arms swing into frame.
        cx1 = int(px1 + 0.40 * w)
        cx2 = int(px1 + 0.60 * w)
        cy1 = int(py1 + 0.22 * h)
        cy2 = int(py1 + 0.45 * h)

    cx1, cx2 = max(0, min(cx1, fw - 1)), max(0, min(cx2, fw - 1))
    cy1, cy2 = max(0, min(cy1, fh - 1)), max(0, min(cy2, fh - 1))

    if cx2 <= cx1 or cy2 <= cy1:
        return frame[max(0, py1):min(py2, fh), max(0, px1):min(px2, fw)]

    # Inward erosion to reject edge bleed from a loose/jittery detection box.
    crop_w, crop_h = cx2 - cx1, cy2 - cy1
    ex = int(crop_w * margin_erosion)
    ey = int(crop_h * margin_erosion)
    ecx1, ecx2 = cx1 + ex, cx2 - ex
    ecy1, ecy2 = cy1 + ey, cy2 - ey

    if ecx2 <= ecx1 or ecy2 <= ecy1:
        return frame[cy1:cy2, cx1:cx2]

    return frame[ecy1:ecy2, ecx1:ecx2]


def _build_valid_pixel_mask(hsv_small):
    """
    Builds a boolean mask over a small HSV image rejecting pixels that are very
    likely skin or background rather than fabric.

    Two rejection criteria, both intentionally conservative (better to keep a
    borderline fabric pixel than to strip too aggressively and starve K-Means):

      1. Skin tone: in OpenCV's H range [0,180], skin across a wide range of tones
         clusters around H in [0, 25] combined with moderate-to-high S and V. This
         is a coarse band, not a full skin-color model, but it's enough to knock
         out the neck/upper-chest strip that the crop geometry can't fully avoid.
      2. Near-zero saturation AND near-max/min value pixels that sit at the very
         extremes (V<10 or V>250 with S<15) are treated as likely sensor clipping
         or background bleed rather than genuine white/black fabric, since real
         fabric under normal lighting rarely hits true 0/255.
    """
    h, s, v = hsv_small[..., 0], hsv_small[..., 1], hsv_small[..., 2]

    skin_mask = (h >= 0) & (h <= 25) & (s >= 40) & (s <= 180) & (v >= 60)
    clipped_mask = ((v < 10) | (v > 250)) & (s < 15)

    reject = skin_mask | clipped_mask
    return ~reject


def get_dominant_color_hsv(crop, k=3, min_valid_fraction=0.15):
    """
    Finds the dominant HSV color of a clothing crop via K-Means, after rejecting
    likely skin/background pixels, and reports an extraction confidence score.

    Args:
        crop (np.ndarray): Cropped clothing region in BGR.
        k (int): Number of K-Means clusters. Raised from 2 to 3 by default so a
            skin-tone remnant, a shadow band, and the true fabric color can each
            form their own cluster instead of two of them being forced together.
        min_valid_fraction (float): If fewer than this fraction of pixels survive
            skin/background rejection, the mask is considered unreliable and the
            function falls back to using all pixels (better than empty input).

    Returns:
        tuple:
            dominant_hsv (np.ndarray): [H, S, V] of the dominant cluster.
            confidence (float): 0.0-1.0 score. Combines (a) the dominant cluster's
                share of valid pixels — a "peaky" dominant cluster means the fabric
                color is consistent — and (b) the fraction of pixels that survived
                skin/background rejection. Low confidence should be treated as a
                signal to down-weight or skip this frame in tracking, not as a
                color to trust equally with a high-confidence read.
    """
    if crop is None or crop.size == 0:
        return np.array([0, 0, 0], dtype=np.float32), 0.0

    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_small = cv2.resize(hsv_crop, (30, 30), interpolation=cv2.INTER_AREA)

    valid_mask = _build_valid_pixel_mask(hsv_small)
    valid_fraction = float(np.mean(valid_mask))

    if valid_fraction >= min_valid_fraction:
        pixels = hsv_small[valid_mask].reshape(-1, 3).astype(np.float32)
    else:
        pixels = hsv_small.reshape(-1, 3).astype(np.float32)
        valid_fraction = 1.0  # mask was unreliable; don't penalize confidence for it

    if pixels.shape[0] < k:
        med = np.median(pixels, axis=0) if pixels.shape[0] > 0 else np.array([0, 0, 0], dtype=np.float32)
        return med, 0.3 * valid_fraction

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    try:
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 6, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        counts = np.bincount(labels, minlength=k)

        dominant_index = int(np.argmax(counts))
        dominant_hsv = centers[dominant_index]
        cluster_purity = float(counts[dominant_index]) / float(len(labels))

        confidence = float(np.clip(0.5 * cluster_purity + 0.5 * valid_fraction, 0.0, 1.0))
        return dominant_hsv, confidence
    except Exception:
        return np.median(pixels, axis=0), 0.3 * valid_fraction


def calculate_color_similarity(hsv1, hsv2):
    """
    Calculates color similarity between two HSV colors (or batches of them).
    Uses circular hue distance and dynamically reweights toward S/V for
    monochrome (black/white/gray) garments, where hue is unstable/meaningless.

    Vectorized: accepts either a single [H,S,V] triple or an (N,3) array for
    batch comparisons (e.g. comparing one detection against all active tracks
    in a real-time pipeline), returning a scalar or an (N,) array respectively.

    Args:
        hsv1 (array-like): [H,S,V] or (N,3).
        hsv2 (array-like): [H,S,V] or (N,3), broadcastable against hsv1.

    Returns:
        float or np.ndarray: Similarity score(s) in [0.0, 1.0].
    """
    a = np.atleast_2d(np.asarray(hsv1, dtype=np.float32))
    b = np.atleast_2d(np.asarray(hsv2, dtype=np.float32))

    h1, s1, v1 = a[:, 0], a[:, 1], a[:, 2]
    h2, s2, v2 = b[:, 0], b[:, 1], b[:, 2]

    dh = np.minimum(np.abs(h1 - h2), 180 - np.abs(h1 - h2))
    dh_norm = dh / 90.0
    ds_norm = np.abs(s1 - s2) / 255.0
    dv_norm = np.abs(v1 - v2) / 255.0

    is_mono1 = (s1 < 40) | (v1 < 40) | ((v1 > 220) & (s1 < 50))
    is_mono2 = (s2 < 40) | (v2 < 40) | ((v2 > 220) & (s2 < 50))
    is_mono = is_mono1 | is_mono2

    w_h = np.where(is_mono, 0.1, 0.6)
    w_s = np.where(is_mono, 0.45, 0.2)
    w_v = np.where(is_mono, 0.45, 0.2)

    distance = w_h * dh_norm + w_s * ds_norm + w_v * dv_norm
    similarity = np.clip(1.0 - distance, 0.0, 1.0)

    return float(similarity[0]) if similarity.shape[0] == 1 else similarity