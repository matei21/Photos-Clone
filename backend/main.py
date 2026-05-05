import uvicorn
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import uuid
import shutil
from typing import List
import numpy as np
import cv2

from db import QdrantDB
from processor import FaceProcessor
from paths import resolve_project_path

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic: DB and Processor are already initialized globally,
    # but we could move them here if needed.
    yield
    # Shutdown logic
    db.close()

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "faces")
DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_faces")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Initialize DB and Processor
db = QdrantDB()
processor = FaceProcessor()

@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    for file in files:
        # 1. Save original to a temporary location to run detection
        temp_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        temp_name = f"temp_{temp_id}{file_extension}"
        temp_path = os.path.join(UPLOAD_DIR, temp_name)
        
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Detect faces and get debug image (original with all boxes)
        faces, debug_img = processor.detect_faces(temp_path)
        
        if debug_img is not None:
            # 3. Save original image with all detections to debug_faces/
            debug_name = f"full_{temp_id}{file_extension}"
            debug_path = os.path.join(DEBUG_DIR, debug_name)
            cv2.imwrite(debug_path, debug_img)
        
        # 4. Process each detected face
        for face_data in faces:
            face_id = str(uuid.uuid4())
            
            # --- CRITICAL: Embed only the crop ---
            embedding = processor.get_embedding(face_data['img'])
            
            # --- CRITICAL: Save only the crop to faces/ ---
            crop_name = f"{face_id}{file_extension}"
            crop_path = os.path.join(UPLOAD_DIR, crop_name)
            cv2.imwrite(crop_path, face_data['img'])
            
            db.add_face(
                face_id=face_id,
                embedding=embedding,
                metadata={
                    "file_name": crop_name, # This is the crop image
                    "original_name": file.filename,
                    "debug_full_image": debug_name if debug_img is not None else None,
                    "cluster": None,
                    "box": face_data['box']
                }
            )
        
        # Cleanup temporary original file
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    return {"message": f"Successfully uploaded {len(files)} files"}

@app.post("/analyze")
async def analyze_clusters():
    faces = db.get_all_faces()
    if not faces:
        return {"message": "No faces found to analyze"}
    
    embeddings = [face.vector for face in faces]
    ids = [face.id for face in faces]
    
    # Perform the sweep from 0.1 to 0.9 with 0.01 step for both linkages
    threshold_values = [round(x * 0.01, 2) for x in range(10, 91)]
    linkages = ['average', 'complete']
    sweep_data = {linkage: {} for linkage in linkages}
    
    for linkage in linkages:
        for threshold in threshold_values:
            labels = processor.cluster_faces(embeddings, distance_threshold=threshold, linkage=linkage)
            
            # Format clusters for this specific threshold and linkage
            clusters = {}
            for face_id, label in zip(ids, labels):
                # Agglomerative clustering doesn't have -1 labels
                c_id = int(label)
                
                if c_id not in clusters:
                    clusters[c_id] = []
                
                # Find the original face metadata for payload
                face_meta = next(f.payload for f in faces if f.id == face_id)
                clusters[c_id].append(face_meta)
                
            # Convert to list format for frontend
            sweep_data[linkage][str(threshold)] = [
                {"cluster_id": c_id, "images": imgs} 
                for c_id, imgs in clusters.items()
            ]
        
    # Also perform the "official" update at a default (e.g., 0.5 threshold, 'average' linkage)
    default_threshold = 0.5
    default_linkage = 'average'
    labels_default = processor.cluster_faces(embeddings, distance_threshold=default_threshold, linkage=default_linkage)
    updates = [(face_id, int(label)) for face_id, label in zip(ids, labels_default)]
    db.update_clusters_batch(updates)
        
    return {
        "message": "Sweep complete",
        "sweep_results": sweep_data,
        "default_threshold": str(default_threshold),
        "default_linkage": default_linkage
    }

@app.get("/clusters")
async def get_clusters():
    clusters = db.get_clusters()
    # Format for frontend
    result = []
    for c_id, faces in clusters.items():
        # Agglomerative clustering has no noise (-1)
        result.append({
            "cluster_id": c_id,
            "images": faces
        })
    return result

@app.get("/db_faces")
async def get_all_faces():
    faces = db.get_all_faces()
    return [{"id": f.id, "payload": f.payload} for f in faces]

@app.delete("/db_faces/{face_id}")
async def delete_face(face_id: str):
    # Get face info before deleting to check for associated file
    all_faces = db.get_all_faces()
    target_face = next((f for f in all_faces if f.id == face_id), None)
    
    db.delete_face(face_id)
    
    if target_face:
        file_name = target_face.payload.get("file_name")
        # Check if any other face uses this file
        remaining_faces = db.get_all_faces()
        still_exists = any(f.payload.get("file_name") == file_name for f in remaining_faces)
        
        if not still_exists:
            file_path = os.path.join(UPLOAD_DIR, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)
                
    return {"message": f"Face {face_id} deleted from database"}

@app.delete("/db_faces")
async def clear_faces():
    db.clear_all()
    # Clear UPLOAD_DIR and DEBUG_DIR to stay in sync
    for directory in [UPLOAD_DIR, DEBUG_DIR]:
        if not os.path.exists(directory): continue
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
    return {"message": "Database and all images cleared"}

app.mount("/faces", StaticFiles(directory=UPLOAD_DIR), name="faces")
app.mount("/debug_faces", StaticFiles(directory=DEBUG_DIR), name="debug_faces")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
