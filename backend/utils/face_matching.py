import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from insightface.app import FaceAnalysis


def _person_box_dims(person_bbox):
    px1, py1, px2, py2 = map(float, person_bbox)
    return px1, py1, px2, py2, (px2 - px1), (py2 - py1)


def compute_face_person_match(face_bbox, person_bbox, buffer_frac=0.22, min_buffer=25, max_buffer=180,
                               horiz_buffer_frac=0.06):
    """
    Scores how well a detected face matches a detected person box, using a buffer
    that scales with the person box's own size rather than a fixed pixel value.

    Why scale-adaptive: a fixed pixel buffer (e.g. 50px) is calibrated for one
    distance from camera. Someone sitting close fills far more of the frame, so
    their person bbox is much taller — a fixed buffer under-corrects and the face
    still clips above the box. Someone farther back has a small person bbox, so
    the same fixed buffer over-corrects and can bleed into a neighboring person's
    space in a crowded scene. Tying the buffer to `person_height` fixes both.

    A small horizontal buffer is also added: close-camera framing (esp. wide-angle
    lenses) tends to push face boxes slightly outside the person box's left/right
    edges too, not just the top, and fast lateral motion causes the same clipping
    horizontally that the vertical buffer was already handling for the top.

    Args:
        face_bbox (list or np.ndarray): [fx1, fy1, fx2, fy2].
        person_bbox (list or np.ndarray): [px1, py1, px2, py2].
        buffer_frac (float): Vertical head buffer as a fraction of person height.
        min_buffer (float): Floor on the vertical buffer in pixels (for small/far people).
        max_buffer (float): Ceiling on the vertical buffer in pixels (avoid runaway
            expansion for very tall/close boxes swallowing neighboring people).
        horiz_buffer_frac (float): Horizontal buffer as a fraction of person width,
            applied to both sides.

    Returns:
        dict with:
            is_contained (bool): True if face center falls in the buffered box AND
                overlap_ratio clears the containment threshold.
            overlap_ratio (float): Intersection area / face area, in [0, 1].
            center_dist_norm (float): Euclidean distance between face center and
                person box center, normalized by the person box diagonal. Used for
                fallback ranking when containment fails for every candidate.
    """
    fx1, fy1, fx2, fy2 = map(float, face_bbox)
    px1, py1, px2, py2, pw, ph = _person_box_dims(person_bbox)

    v_buffer = float(np.clip(buffer_frac * ph, min_buffer, max_buffer))
    h_buffer = horiz_buffer_frac * pw

    bpx1 = px1 - h_buffer
    bpx2 = px2 + h_buffer
    bpy1 = max(0.0, py1 - v_buffer)
    bpy2 = py2

    fc_x = (fx1 + fx2) / 2.0
    fc_y = (fy1 + fy2) / 2.0
    pc_x = (px1 + px2) / 2.0
    pc_y = (py1 + py2) / 2.0
    diag = max(1.0, np.hypot(pw, ph))
    center_dist_norm = float(np.hypot(fc_x - pc_x, fc_y - pc_y) / diag)

    overlap_ratio = 0.0
    is_contained = False
    if bpx1 <= fc_x <= bpx2 and bpy1 <= fc_y <= bpy2:
        ix1, iy1 = max(bpx1, fx1), max(bpy1, fy1)
        ix2, iy2 = min(bpx2, fx2), min(bpy2, fy2)
        if ix2 > ix1 and iy2 > iy1:
            intersect_area = (ix2 - ix1) * (iy2 - iy1)
            face_area = (fx2 - fx1) * (fy2 - fy1)
            overlap_ratio = intersect_area / face_area if face_area > 0 else 0.0
            is_contained = overlap_ratio >= 0.5

    return {
        "is_contained": is_contained,
        "overlap_ratio": overlap_ratio,
        "center_dist_norm": center_dist_norm,
    }


# Kept for backward compatibility with existing call sites.
def is_face_in_person_box(face_bbox, person_bbox, head_buffer=50, overlap_thresh=0.5):
    """Legacy fixed-buffer check. Prefer compute_face_person_match for new code."""
    result = compute_face_person_match(face_bbox, person_bbox)
    return result["is_contained"], result["overlap_ratio"]


class FaceMatcher:
    def __init__(self, model_name='buffalo_l', det_size=(1280, 1280)):
        """
        Initializes the InsightFace FaceAnalysis application.
        Automatically detects whether PyTorch has access to a CUDA-enabled GPU.

        Args:
            model_name (str): The pre-trained model pack to use (default: 'buffalo_l').
            det_size (tuple): The detection resolution for the face detector.
        """
        print("[INFO] Initializing InsightFace model...")
        ctx_id = 0 if torch.cuda.is_available() else -1
        if ctx_id == 0:
            print("[INFO] GPU detected. Running face matching on CUDA.")
        else:
            print("[INFO] No GPU detected. Running face matching on CPU.")

        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        print("[INFO] InsightFace initialization complete.")

    def get_face_embedding(self, img):
        """
        Detects faces in the input image and returns the embedding of the largest face.

        Args:
            img (np.ndarray): Reference BGR image.

        Returns:
            np.ndarray or None: The normalized 512-D face embedding vector, or None if no face is found.
        """
        if img is None or img.size == 0:
            return None

        faces = self.app.get(img)
        if len(faces) == 0:
            return None

        faces = sorted(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        target_face = faces[0]

        if hasattr(target_face, 'normed_embedding') and target_face.normed_embedding is not None:
            return target_face.normed_embedding
        elif hasattr(target_face, 'embedding') and target_face.embedding is not None:
            emb = target_face.embedding
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb

        return None

    def match_faces_to_people(self, people_boxes, frame_faces, fallback_max_center_dist=0.65):
        """
        Maps face detections to person bounding boxes.

        Two-stage strategy:

          1. Optimal assignment on strict containment. Every (face, person) pair
             that passes `compute_face_person_match`'s containment test is scored
             by 1 - overlap_ratio and solved with Hungarian assignment, instead of
             greedily assigning the first face that meets threshold to each person.
             Greedy assignment lets a mediocre early match block a better-fitting
             face from ever reaching that person index — this matters more, not
             less, in crowded/close-camera scenes where boxes are large and touch.

          2. Fallback pass for leftover faces. Any face that didn't clear strict
             containment for *any* person (the actual "clipped box from fast
             motion / close camera" failure) is instead matched to its nearest
             remaining person by normalized center distance, as long as that
             distance is under `fallback_max_center_dist`. This is a deliberately
             lower-confidence match — it's flagged as such — so downstream logic
             (e.g. re-ID gating) can choose to trust it less than a strict match
             rather than treating a fallback the same as a confirmed containment.

        Args:
            people_boxes (list of list): [[x1, y1, x2, y2], ...] from YOLO.
            frame_faces (list): Detected Face objects from InsightFace.
            fallback_max_center_dist (float): Max normalized center distance
                (fraction of person box diagonal) allowed for a fallback match.
                Set to None to disable the fallback stage entirely.

        Returns:
            dict: person_index -> {
                "face": Face object,
                "match_type": "contained" | "fallback",
                "confidence": float in [0, 1],
            }
        """
        mapping = {}
        if not frame_faces or len(people_boxes) == 0:
            return mapping

        n_faces, n_people = len(frame_faces), len(people_boxes)
        overlap_matrix = np.zeros((n_faces, n_people), dtype=np.float32)
        contained_matrix = np.zeros((n_faces, n_people), dtype=bool)
        dist_matrix = np.full((n_faces, n_people), np.inf, dtype=np.float32)

        for i, face in enumerate(frame_faces):
            for j, pb in enumerate(people_boxes):
                res = compute_face_person_match(face.bbox, pb)
                overlap_matrix[i, j] = res["overlap_ratio"]
                contained_matrix[i, j] = res["is_contained"]
                dist_matrix[i, j] = res["center_dist_norm"]

        # --- Stage 1: optimal assignment restricted to strictly-contained pairs ---
        cost = np.where(contained_matrix, 1.0 - overlap_matrix, 1e6)
        if np.any(contained_matrix):
            row_idx, col_idx = linear_sum_assignment(cost)
            for i, j in zip(row_idx, col_idx):
                if contained_matrix[i, j]:
                    mapping[j] = {
                        "face": frame_faces[i],
                        "match_type": "contained",
                        "confidence": float(overlap_matrix[i, j]),
                    }

        matched_face_idx = {
            i for i, face in enumerate(frame_faces)
            for j in mapping
            if mapping[j]["face"] is face
        }
        matched_person_idx = set(mapping.keys())

        # --- Stage 2: fallback fail-safe for faces with no valid containment match ---
        if fallback_max_center_dist is not None:
            for i in range(n_faces):
                if i in matched_face_idx:
                    continue
                remaining_people = [j for j in range(n_people) if j not in matched_person_idx]
                if not remaining_people:
                    continue
                dists = [(j, dist_matrix[i, j]) for j in remaining_people]
                best_j, best_dist = min(dists, key=lambda t: t[1])
                if best_dist <= fallback_max_center_dist:
                    # Confidence decays linearly to 0 at the max allowed distance.
                    confidence = float(max(0.0, 1.0 - (best_dist / fallback_max_center_dist))) * 0.6
                    mapping[best_j] = {
                        "face": frame_faces[i],
                        "match_type": "fallback",
                        "confidence": confidence,
                    }
                    matched_person_idx.add(best_j)

        return mapping


def calculate_face_similarity(emb1, emb2):
    """
    Computes the cosine similarity between two face embeddings.
    Since both embeddings are normalized, this is mathematically identical to their dot product.

    Args:
        emb1 (np.ndarray): Target face embedding vector.
        emb2 (np.ndarray): Candidate face embedding vector.

    Returns:
        float: Cosine similarity score between -1.0 and 1.0 (matching faces score higher, typically > 0.40).
    """
    if emb1 is None or emb2 is None:
        return 0.0
    return float(np.dot(emb1, emb2))