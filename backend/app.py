import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

# Import local utility modules
from utils.color_matching import extract_clothing_region, get_dominant_color_hsv, calculate_color_similarity
from utils.face_matching import FaceMatcher, calculate_face_similarity

def resize_image_if_large(img, max_dim=1024):
    """
    Resizes an image dynamically if its width or height exceeds max_dim.
    Maintains the aspect ratio to prevent memory crashes and speed up inference.
    """
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        print(f"[INFO] Resizing target image from {w}x{h} to {new_w}x{new_h} to avoid memory overhead.")
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

def robust_imread(path, flags=cv2.IMREAD_COLOR):
    """
    Reads an image using standard Python byte reading to bypass issues with
    ultralytics monkey-patched cv2.imread on Windows systems.
    """
    with open(path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, flags)

def get_target_clothing_color(yolo_model, target_img, target_face_bbox=None, device="cpu"):
    """
    Detects the main person in the target image and extracts their dominant clothing color.
    Falls back to the entire image if no person is detected.
    """
    results = yolo_model(target_img, classes=[0], device=device, verbose=False)
    boxes = results[0].boxes
    
    if len(boxes) > 0:
        # Find the largest detected person box
        largest_box = None
        max_area = 0
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area > max_area:
                max_area = area
                largest_box = xyxy
        
        # Extract clothing region and dominant color
        clothing_crop = extract_clothing_region(target_img, largest_box, face_bbox=target_face_bbox)
        dominant_hsv, confidence = get_dominant_color_hsv(clothing_crop)
        print(f"[INFO] Target person detected. Dominant clothing HSV: {dominant_hsv} (Conf: {confidence:.2f})")
        return dominant_hsv
    else:
        # Fallback: Extract dominant color from the lower 2/3rds of the image (assuming portrait/torso shot)
        h, w = target_img.shape[:2]
        fallback_crop = target_img[int(h*0.3):int(h*0.9), int(w*0.1):int(w*0.9)]
        dominant_hsv, confidence = get_dominant_color_hsv(fallback_crop)
        print(f"[WARN] No person bounding box found in target image. Using fallback region. Dominant HSV: {dominant_hsv} (Conf: {confidence:.2f})")
        return dominant_hsv

def main():
    parser = argparse.ArgumentParser(description="SafeTrack AI - Live Surveillance & Tracking Backend")
    parser.add_argument(
        "--target", 
        type=str, 
        default=os.path.join("storage", "uploads", "target.jpg"),
        help="Path to the target reference photo (e.g. target.jpg)"
    )
    parser.add_argument(
        "--webcam", 
        type=int, 
        default=0, 
        help="Webcam camera index (default: 0)"
    )
    parser.add_argument(
        "--color_thresh", 
        type=float, 
        default=50.0, 
        help="Minimum similarity score for clothing color match in percent (default: 50.0)"
    )
    parser.add_argument(
        "--face_thresh", 
        type=float, 
        default=45.0, 
        help="Minimum cosine similarity score for face verification in percent (default: 45.0)"
    )
    parser.add_argument(
        "--color_name", 
        type=str, 
        default="none", 
        help="Optional manual clothing color name override (e.g. red, blue, green, yellow, black, white, gray)"
    )
    args = parser.parse_args()

    # Convert threshold percentages (0-100) to standard similarity ratios (0-1) for comparison
    color_thresh_ratio = args.color_thresh / 100.0
    face_thresh_ratio = args.face_thresh / 100.0

    # Create the storage uploads directory if it does not exist
    os.makedirs(os.path.dirname(args.target), exist_ok=True)

    print("=========================================")
    print("        SAFETRACK AI SURVEILLANCE        ")
    print("=========================================")
    
    # 1. Load YOLOv8 for real-time person detection
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Running YOLOv8 on device: {device.upper()}")
    print("[INFO] Loading YOLOv8 nano model...")
    yolo_model = YOLO("yolov8n.pt")
    
    # 2. Load InsightFace Face Matcher (initialize with 640x640 for target image registration)
    face_matcher = FaceMatcher(det_size=(640, 640))
    
    # 3. Read and preprocess the Target reference image
    target_path = args.target
    if not os.path.exists(target_path):
        # Auto-resolve directory context (e.g. running from inside backend/ directory)
        parent_candidate = os.path.join("..", target_path)
        if os.path.exists(parent_candidate):
            target_path = parent_candidate
        else:
            # Check if running from root and target path was provided relative to root
            backend_candidate = os.path.join("backend", target_path)
            if os.path.exists(backend_candidate):
                target_path = backend_candidate
    target_path = os.path.abspath(target_path)

    if not os.path.exists(target_path):
        print(f"[ERROR] Target photo not found at: '{args.target}'")
        print("[HELP] Please upload a reference image named 'target.jpg' to storage/uploads/")
        print("[HELP] Or specify another path using: python backend/app.py --target path/to/image.jpg")
        return

    print(f"[INFO] Reading target image from: {target_path}")
    target_img = robust_imread(target_path)
    if target_img is None:
        print("[ERROR] Failed to read target image file.")
        return
        
    # Resize to prevent crashes if resolution is extremely high
    target_img = resize_image_if_large(target_img)
    
    # Extract Target Features (Priority 1 & 2)
    print("[INFO] Extracting target features...")
    
    # Run face detection on the target image
    target_faces = face_matcher.app.get(target_img)
    
    # Lenient Face Detection: If initial extraction fails, attempt image enhancements (CLAHE and Upscaling)
    if len(target_faces) == 0:
        print("[WARN] No face could be detected in target image on first pass. Attempting CLAHE contrast enhancement...")
        # Apply CLAHE to resolve backlit / shadows
        lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        target_faces = face_matcher.app.get(enhanced_img)
        
        if len(target_faces) == 0:
            print("[WARN] Still no face detected. Attempting 2x resolution upscale fallback...")
            # Upscale image using cubic interpolation for smaller/turned faces
            upscaled_img = cv2.resize(target_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            target_faces = face_matcher.app.get(upscaled_img)
            
    # Extract bounding box and embedding of the largest detected target face
    target_face_bbox = None
    target_face_embedding = None
    
    if len(target_faces) > 0:
        # Sort faces by size in descending order
        target_faces_sorted = sorted(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        largest_face = target_faces_sorted[0]
        target_face_bbox = largest_face.bbox
        
        # Extract embedding
        if hasattr(largest_face, 'normed_embedding') and largest_face.normed_embedding is not None:
            target_face_embedding = largest_face.normed_embedding
        elif hasattr(largest_face, 'embedding') and largest_face.embedding is not None:
            emb = largest_face.embedding
            norm = np.linalg.norm(emb)
            target_face_embedding = emb / norm if norm > 0 else emb
            
    # Text-to-HSV mapping dictionary for common colors
    COLOR_NAME_TO_HSV = {
        "red": [0.0, 200.0, 200.0],
        "blue": [110.0, 200.0, 200.0],
        "green": [60.0, 200.0, 200.0],
        "yellow": [30.0, 200.0, 200.0],
        "orange": [15.0, 200.0, 200.0],
        "purple": [145.0, 200.0, 200.0],
        "black": [0.0, 0.0, 15.0],
        "white": [0.0, 0.0, 240.0],
        "gray": [0.0, 0.0, 120.0],
        "grey": [0.0, 0.0, 120.0]
    }
    
    target_color_hsv = None
    # 3. Override Condition: Check if manual color string is provided and valid
    if args.color_name.lower() != "none":
        color_key = args.color_name.lower()
        if color_key in COLOR_NAME_TO_HSV:
            target_color_hsv = np.array(COLOR_NAME_TO_HSV[color_key], dtype=np.float32)
            print(f"[INFO] Using manual target clothing color override: '{color_key}' -> HSV: {target_color_hsv}")
        else:
            print(f"[WARN] Unknown manual color '{args.color_name}'. Falling back to target image clothing extraction.")
            
    if target_color_hsv is None:
        # Extract target clothing color from the reference photo
        target_color_hsv = get_target_clothing_color(yolo_model, target_img, target_face_bbox=target_face_bbox, device=device)
    
    if target_face_embedding is None:
        print("[WARN] No face could be detected in target image after fallbacks. Matching will rely only on clothing color similarity.")
        # Dynamic Threshold Fallback: Lower color threshold to 45.0% (0.45 ratio) internally
        print("[INFO] Fallback activated: Lowering color threshold internally to 45.0% (ratio 0.45)")
        color_thresh_ratio = 0.45
    else:
        print("[INFO] Target face embedding successfully registered.")

    # Re-prepare the face analysis detector to (1280, 1280) for high-resolution live streaming detection
    print("[INFO] Scaling face detector input resolution to 1280x1280 for live tracking...")
    ctx_id = 0 if torch.cuda.is_available() else -1
    face_matcher.app.prepare(ctx_id=ctx_id, det_size=(1280, 1280))

    # 4. Open webcam dynamically
    print(f"[INFO] Opening webcam (index {args.webcam})...")
    cap = cv2.VideoCapture(args.webcam)
    if not cap.isOpened():
        print(f"[ERROR] Could not open webcam at index {args.webcam}. Check connection and permissions.")
        return

    print("[SUCCESS] Pipeline ready. Press 'q' inside the video window to quit.")
    print("=========================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from webcam. Retrying...")
            continue
            
        # Draw target reference status at the top-left of the display frame
        cv2.putText(
            frame, 
            "SafeTrack AI - Active Surveillance", 
            (15, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (0, 255, 255), 
            2, 
            cv2.LINE_AA
        )

        # A. Detect people in the frame using YOLOv8 (class 0 is person)
        results = yolo_model(frame, classes=[0], device=device, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # B. Real-Time Dynamic Face Illumination (Local CLAHE for dark/backlit faces)
        enhanced_frame = frame.copy()
        for box in boxes:
            px1, py1, px2, py2 = map(int, box)
            
            # Ensure coordinates are within frame bounds
            fh, fw = frame.shape[:2]
            px1 = max(0, min(px1, fw - 1))
            px2 = max(0, min(px2, fw - 1))
            py1 = max(0, min(py1, fh - 1))
            py2 = max(0, min(py2, fh - 1))
            
            w = px2 - px1
            h = py2 - py1
            if w <= 0 or h <= 0:
                continue
                
            # Focus on the head/face region (typically upper 45% of person height)
            hy2 = int(py1 + 0.45 * h)
            hy2 = max(py1 + 1, min(hy2, py2))
            
            head_crop = frame[py1:hy2, px1:px2]
            if head_crop.size == 0:
                continue
                
            # Check if region is dark/backlit using lightness channel in LAB space
            lab = cv2.cvtColor(head_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_l = np.mean(l)
            
            if mean_l < 115:  # Lightness threshold to detect shadows/backlighting
                # Apply CLAHE to resolve shadows on the face dynamically
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_head = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                enhanced_frame[py1:hy2, px1:px2] = enhanced_head
        
        # C. Run face detection & embedding extraction on the enhanced frame
        frame_faces = face_matcher.app.get(enhanced_frame)
        
        # D. Map detected faces to person boxes based on bounding box overlap
        face_to_person_map = face_matcher.match_faces_to_people(boxes, frame_faces)

        # E. Process each detected person
        for idx, person_box in enumerate(boxes):
            px1, py1, px2, py2 = map(int, person_box)
            
            # 1. Dual Bounding Box Filters (Area + Aspect Ratio)
            box_width = px2 - px1
            box_height = py2 - py1
            box_area = box_width * box_height
            aspect_ratio = box_width / float(box_height) if box_height > 0 else 0.0
            
            is_target = False
            is_potential = False
            confidence_pct = 0.0
            color_sim = 0.0
            face_sim = 0.0
            face_sim_str = "N/A"
            
            # Human Shape Rule: wider boxes (ratio > 1.6, like stretched arm) or small boxes (area < 40000) are bypassed
            if box_area < 40000 or aspect_ratio > 1.6:
                face_sim_str = "N/A (Filtered Shape)"
            else:
                # Query face coordinates to center the torso crop below the chin if available
                associated_face_info = face_to_person_map.get(idx)
                associated_face = associated_face_info["face"] if associated_face_info is not None else None
                face_bbox = associated_face.bbox if associated_face is not None else None
                
                # Priority 1: Clothing Color Similarity
                clothing_crop = extract_clothing_region(frame, person_box, face_bbox=face_bbox)
                dominant_hsv, color_conf = get_dominant_color_hsv(clothing_crop)
                color_sim = calculate_color_similarity(target_color_hsv, dominant_hsv)
                
                if target_face_embedding is not None and associated_face is not None:
                    # Priority 2: Face Verification
                    face_sim = calculate_face_similarity(target_face_embedding, associated_face.normed_embedding)
                    face_sim_str = f"{face_sim * 100.0:.1f}%"
                    
                    # 1. Biometric Override: If face similarity is extremely high (>= 58.0%),
                    # we confirm the target immediately, even if they changed their clothes
                    if face_sim >= 0.58:
                        is_target = True
                        confidence_pct = 70.0 + ((face_sim - 0.58) / 0.27) * 28.0
                        confidence_pct = min(99.9, max(70.0, confidence_pct))
                    # 2. Combined Match: Passes both color and face thresholds
                    elif color_sim >= color_thresh_ratio and face_sim >= face_thresh_ratio:
                        is_target = True
                        # Calibrate face similarity to a readable confidence percentage
                        denom = 0.85 - face_thresh_ratio
                        if denom <= 0:
                            denom = 0.01
                        confidence_pct = 70.0 + ((face_sim - face_thresh_ratio) / denom) * 28.0
                        confidence_pct = min(99.9, max(70.0, confidence_pct))
                    elif color_sim >= color_thresh_ratio:
                        # Color similarity matches but face similarity doesn't pass face threshold
                        is_potential = True
                elif target_face_embedding is None:
                    # Fallback to clothing-only match if no face embedding was registered for the target
                    if color_sim >= color_thresh_ratio:
                        is_target = True
                        confidence_pct = color_sim * 100.0
                else:
                    # Target face embedding is registered, but no face detected in the frame for this person
                    if color_sim >= color_thresh_ratio:
                        is_potential = True

            # Print debug console log for every detected person box
            print(f"[DEBUG] Person Box: Color Similarity: {color_sim * 100.0:.1f}%, Face Similarity: {face_sim_str}")

            # F. Annotate bounding boxes based on match outcome
            if is_target:
                # Target Matched: Bright Green Box
                box_color = (0, 255, 0)
                label = f"TARGET MATCHED [{confidence_pct:.1f}%]"
            elif is_potential:
                # Potential Target: Green/Orange (Amber) Box
                box_color = (0, 128, 255)
                label = f"POTENTIAL TARGET [Color Sim: {color_sim * 100.0:.1f}%]"
            else:
                # Standard Person: Blue Box
                box_color = (255, 0, 0)
                label = f"Person (C:{color_sim * 100.0:.1f}%)"

            # Draw bounding box
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            
            # Draw label with background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Ensure text box is within window limits
            text_y = max(py1, text_h + 10)
            cv2.rectangle(frame, (px1, text_y - text_h - 10), (px1 + text_w + 10, text_y), box_color, -1)
            cv2.putText(frame, label, (px1 + 5, text_y - 5), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Show the video feed
        cv2.imshow("SafeTrack AI - Backend Surveillance Feed", frame)
        
        # Exit feed on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera released and windows closed. SafeTrack AI shut down cleanly.")

if __name__ == "__main__":
    main()
