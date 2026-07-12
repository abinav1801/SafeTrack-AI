# SafeTrack AI – Intelligent Missing Child Tracking System

SafeTrack AI is an automated video surveillance pipeline engineered to locate specific targets in live webcam feeds or CCTV streams. The system leverages a dual-priority approach, utilizing facial embeddings when available, and dynamically falling back to precise clothing color matching under poor lighting conditions, face clipping, or partial occlusions.

## 🚀 Key Features

*   **Dual-Priority Verification:** Attempts face matching using InsightFace; automatically cascades to clothing-based color tracking if a target face cannot be resolved.
*   **Manual Dress Color Override:** Supports entering specific garment colors via the command-line interface (`--color_name`), mapping text inputs directly to the standard OpenCV HSV space and bypassing image-based color extraction.
*   **Robust False-Positive Suppression:**
    *   *Bounding Box Area Filter:* Automatically drops detections smaller than 40,000 pixels or 15% of the frame to filter out stray hands/objects.
    *   *Human Aspect Ratio Filter:* Enforces structural parameters (`aspect_ratio <= 1.2`) to immediately filter out horizontal shapes like outstretched arms or legs.
*   **Strict Center-Chest Anchoring:** Samples clothing colors exclusively from a tight vertical strip at the upper torso (40% to 60% width), preventing arm movements or backgrounds from polluting the metrics.
*   **Lenient Image Pre-Processing:** Integrates dynamic CLAHE contrast adjustment and 2x cubic upscaling fallbacks during initialization to extract faces under heavy backlighting or shadow.

## 🛠️ Prerequisites & Stack

- **Backend:** Python 3.10+, OpenCV (cv2), NumPy, Torch, Ultralytics YOLOv8, InsightFace (Buffalo_L Model Pack)
- **Frontend Framework:** Next.js / React.js or Standard HTML5/TailwindCSS Boilerplate
- **State Management / Communication:** Axios (HTTP API calls to backend) & WebSocket (for real-time live feed frames)

## 📦 Project Structure

```text
├── backend/
│   ├── app.py                     # Main execution loop & CLI coordinator
│   └── utils/
│       ├── color_matching.py      # Torso isolation, K-Means, and HSV similarity
│       └── face_matching.py       # Spatial mapping & InsightFace operations
├── frontend/                      # Blueprint for UI Dashboard
│   ├── public/
│   │   └── assets/                # Logos and fallback surveillance graphics
│   ├── src/
│   │   ├── components/
│   │   │   ├── TargetForm.jsx     # Form to upload child target image & input color name
│   │   │   ├── VideoStream.jsx    # Real-time WebRTC/WebSocket player canvas for feed
│   │   │   └── MetricsPanel.jsx   # Live log viewer showing HSV matching % and state
│   │   ├── App.jsx                # Main interface container layout
│   │   └── index.css              # Custom TailwindCSS styling overrides
│   └── package.json               # Frontend dependencies & start scripts
└── storage/
    └── uploads/
        └── target.jpg             # Reference photo uploaded via user interface