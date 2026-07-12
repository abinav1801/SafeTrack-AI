import os
import threading
import cv2
import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

# Import local utility modules
from utils.color_matching import extract_clothing_region, get_dominant_color_hsv, calculate_color_similarity
from utils.face_matching import FaceMatcher, calculate_face_similarity

app = FastAPI(title="SafeTrack AI - Live Web API Server")

# Enable CORS for cross-origin React frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store active search target state
target_face_embedding = None
target_color_hsv = None
color_thresh_ratio = 0.50
face_thresh_ratio = 0.45
color_override_name = "none"

surveillance_thread = None
is_running = False
yolo_model = None
face_matcher = None

# Custom image reading helper to bypass Windows file-lock/Monkey-patch issues
def robust_imread(path, flags=cv2.IMREAD_COLOR):
    with open(path, 'rb') as f:
        file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
    return cv2.imdecode(file_bytes, flags)

def get_target_clothing_color(yolo_model, target_img, target_face_bbox=None):
    results = yolo_model(target_img, classes=[0], device="cpu", verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        largest_box = None
        max_area = 0
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if area > max_area:
                max_area = area
                largest_box = xyxy
        clothing_crop = extract_clothing_region(target_img, largest_box, face_bbox=target_face_bbox)
        dominant_hsv, confidence = get_dominant_color_hsv(clothing_crop)
        return dominant_hsv
    else:
        h, w = target_img.shape[:2]
        fallback_crop = target_img[int(h*0.3):int(h*0.9), int(w*0.1):int(w*0.9)]
        dominant_hsv, confidence = get_dominant_color_hsv(fallback_crop)
        return dominant_hsv

def resize_image_if_large(img, max_dim=1024):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

@app.on_event("startup")
def load_models():
    global yolo_model, face_matcher
    print("[INFO] Server starting up. Pre-loading models...")
    yolo_model = YOLO("yolov8n.pt")
    face_matcher = FaceMatcher(det_size=(640, 640))
    print("[SUCCESS] Server initialization complete. Ready for API requests.")

@app.post("/api/register")
async def register_target(
    file: UploadFile = File(...),
    color_name: str = Form("none")
):
    global target_face_embedding, target_color_hsv, color_override_name, color_thresh_ratio
    
    # Ensure uploads folder exists
    upload_dir = os.path.join("storage", "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    target_path = os.path.abspath(os.path.join(upload_dir, "target.jpg"))
    
    # Save the uploaded file bytes
    with open(target_path, "wb") as buffer:
        buffer.write(await file.read())
        
    print(f"[INFO] Uploaded image successfully written to: {target_path}")
    
    target_img = robust_imread(target_path)
    if target_img is None:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image format.")
        
    target_img = resize_image_if_large(target_img)
    
    target_face_bbox = None
    target_face_embedding = None
    color_thresh_ratio = 0.50  # Reset
    color_override_name = color_name
    
    # 1. Target Face Verification (at 640x640)
    target_faces = face_matcher.app.get(target_img)
    if len(target_faces) == 0:
        # Fallback 1: CLAHE enhancement
        lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
        target_faces = face_matcher.app.get(enhanced_img)
        
        if len(target_faces) == 0:
            # Fallback 2: 2x resolution upscale
            upscaled_img = cv2.resize(target_img, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            target_faces = face_matcher.app.get(upscaled_img)
            
    if len(target_faces) > 0:
        target_faces_sorted = sorted(target_faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)
        largest_face = target_faces_sorted[0]
        target_face_bbox = largest_face.bbox
        
        if hasattr(largest_face, 'normed_embedding') and largest_face.normed_embedding is not None:
            target_face_embedding = largest_face.normed_embedding
        elif hasattr(largest_face, 'embedding') and largest_face.embedding is not None:
            emb = largest_face.embedding
            norm = np.linalg.norm(emb)
            target_face_embedding = emb / norm if norm > 0 else emb

    # 2. Target Clothing Color Extraction
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
    
    is_overridden = False
    if color_name.lower() != "none" and color_name.lower() in COLOR_NAME_TO_HSV:
        target_color_hsv = np.array(COLOR_NAME_TO_HSV[color_name.lower()], dtype=np.float32)
        is_overridden = True
        print(f"[INFO] Using manual override color: {color_name} -> HSV: {target_color_hsv}")
    else:
        target_color_hsv = get_target_clothing_color(yolo_model, target_img, target_face_bbox=target_face_bbox)
        print(f"[INFO] Extracted color from photo -> HSV: {target_color_hsv}")
        
    face_detected = target_face_embedding is not None
    if not face_detected:
        print("[WARN] Target photo has no face. Using clothing-only fallback matching (color thresh: 45.0%).")
        color_thresh_ratio = 0.45
        
    return {
        "status": "success",
        "face_detected": face_detected,
        "is_color_overridden": is_overridden,
        "color_override_name": color_override_name,
        "dominant_color_hsv": target_color_hsv.tolist() if target_color_hsv is not None else None
    }

def run_surveillance():
    global is_running, face_matcher, yolo_model, target_face_embedding, target_color_hsv, color_thresh_ratio, face_thresh_ratio
    
    print("[INFO] Re-configuring face detector resolution to 1280x1280 for webcam surveillance...")
    ctx_id = 0 if torch.cuda.is_available() else -1
    face_matcher.app.prepare(ctx_id=ctx_id, det_size=(1280, 1280))
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam device index 0. Thread exiting.")
        is_running = False
        return
        
    print("[SUCCESS] Surveillance capture active. Processing camera frames.")
    
    while cap.isOpened() and is_running:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab camera frame.")
            break
            
        enhanced_frame = frame.copy()
        
        # A. Detect all people inside the frame using YOLOv8
        results = yolo_model(frame, classes=[0], device="cpu", verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # B. Local CLAHE enhancement for backlit faces
        for idx, person_box in enumerate(boxes):
            px1, py1, px2, py2 = map(int, person_box)
            h = py2 - py1
            hy2 = int(py1 + 0.45 * h)
            hy2 = max(py1 + 1, min(hy2, py2))
            
            head_crop = frame[py1:hy2, px1:px2]
            if head_crop.size == 0:
                continue
                
            lab = cv2.cvtColor(head_crop, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            mean_l = np.mean(l)
            
            if mean_l < 115:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                enhanced_head = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                enhanced_frame[py1:hy2, px1:px2] = enhanced_head
                
        # C. Detect faces in enhanced frame & map them to person bounding boxes
        frame_faces = face_matcher.app.get(enhanced_frame)
        face_to_person_map = face_matcher.match_faces_to_people(boxes, frame_faces)
        
        # D. Match prioritizations
        for idx, person_box in enumerate(boxes):
            px1, py1, px2, py2 = map(int, person_box)
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
            
            # Human Bounding Box Constraints
            if box_area < 40000 or aspect_ratio > 1.6:
                face_sim_str = "N/A (Filtered Shape)"
            else:
                associated_face_info = face_to_person_map.get(idx)
                associated_face = associated_face_info["face"] if associated_face_info is not None else None
                face_bbox = associated_face.bbox if associated_face is not None else None
                
                # Priority 1: Torso clothing HSV crop and matching
                clothing_crop = extract_clothing_region(frame, person_box, face_bbox=face_bbox)
                dominant_hsv, color_conf = get_dominant_color_hsv(clothing_crop)
                
                if target_color_hsv is not None:
                    color_sim = calculate_color_similarity(target_color_hsv, dominant_hsv)
                    
                # Priority 2: Biometric verification
                if target_face_embedding is not None and associated_face is not None:
                    face_sim = calculate_face_similarity(target_face_embedding, associated_face.normed_embedding)
                    face_sim_str = f"{face_sim * 100.0:.1f}%"
                    
                    # Match Decision Logic
                    if face_sim >= 0.58:  # Biometric override bypass
                        is_target = True
                        confidence_pct = 70.0 + ((face_sim - 0.58) / 0.27) * 28.0
                        confidence_pct = min(99.9, max(70.0, confidence_pct))
                    elif color_sim >= color_thresh_ratio and face_sim >= face_thresh_ratio:
                        is_target = True
                        denom = 0.85 - face_thresh_ratio
                        if denom <= 0:
                            denom = 0.01
                        confidence_pct = 70.0 + ((face_sim - face_thresh_ratio) / denom) * 28.0
                        confidence_pct = min(99.9, max(70.0, confidence_pct))
                    elif color_sim >= color_thresh_ratio:
                        is_potential = True
                elif target_face_embedding is None:
                    if color_sim >= color_thresh_ratio:
                        is_target = True
                        confidence_pct = color_sim * 100.0
                        
            print(f"[DEBUG] Person Box: Color Similarity: {color_sim * 100.0:.1f}%, Face Similarity: {face_sim_str}")
            
            # E. Annotate bounding boxes based on match outcome
            if is_target:
                box_color = (0, 255, 0)
                label = f"TARGET MATCHED [{confidence_pct:.1f}%]"
            elif is_potential:
                box_color = (0, 128, 255)
                label = f"POTENTIAL TARGET [Color Sim: {color_sim * 100.0:.1f}%]"
            else:
                box_color = (255, 0, 0)
                label = f"Person (C:{color_sim * 100.0:.1f}%)"
                
            cv2.rectangle(frame, (px1, py1), (px2, py2), box_color, 2)
            cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
            
        cv2.imshow("SafeTrack AI - CCTV Surveillance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    is_running = False
    print("[INFO] Webcam surveillance feed released.")

@app.post("/api/start")
async def start_feed():
    global is_running, surveillance_thread, target_color_hsv
    if target_color_hsv is None:
        raise HTTPException(status_code=400, detail="No target profile registered. Please register target first.")
        
    if is_running:
        return {"status": "already_running", "message": "Surveillance feed is already active."}
        
    is_running = True
    surveillance_thread = threading.Thread(target=run_surveillance, daemon=True)
    surveillance_thread.start()
    return {"status": "success", "message": "Live webcam surveillance feed initialized."}

@app.post("/api/stop")
async def stop_feed():
    global is_running
    if not is_running:
        return {"status": "not_running", "message": "Surveillance is not active."}
        
    is_running = False
    return {"status": "success", "message": "Halting camera loop..."}

@app.get("/api/status")
async def get_status():
    global is_running, target_face_embedding, target_color_hsv
    return {
        "is_running": is_running,
        "is_target_registered": target_color_hsv is not None,
        "face_detected": target_face_embedding is not None,
        "color_override": color_override_name
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
