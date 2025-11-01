from fastapi import FastAPI
from pydantic import BaseModel
import torch
import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI(title="BioLens WBC Detection API")

# ----------------------------
# Load YOLOv11 model
# ----------------------------
# Change the repo to yolov11 if your model was trained using YOLOv11
# If it's trained via ultralytics (newer YOLO versions), use `from ultralytics import YOLO`
from ultralytics import YOLO

# Load your trained model (make sure 'model.pt' is in the same directory)
model = YOLO("model.pt")

IMG_SIZE = 832  # resize target
CONFIDENCE_THRESHOLD = 0.95  # only detections above 95%

class ImageRequest(BaseModel):
    image_url: str

@app.post("/predict")
async def predict(data: ImageRequest):
    try:
        # Fetch image from the provided URL
        response = requests.get(data.image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Convert to NumPy array and resize
        img_cv = np.array(img)
        img_resized = cv2.resize(img_cv, (IMG_SIZE, IMG_SIZE))

        # Run YOLOv11 inference
        results = model.predict(source=img_resized, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = model.names[cls]
                detections.append({
                    "cell_type": label,
                    "confidence": round(conf, 3),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

        return {"status": "success", "detections": detections}

    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/")
def root():
    return {"message": "BioLens WBC Model API running successfully 🚀"}
