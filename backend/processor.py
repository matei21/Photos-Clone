import torch
import cv2
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from models_arch import FullFaceNet
from paths import resolve_project_path

class FaceProcessor:
    def __init__(self, arcface_path='models/arcface.pth'):
        arcface_path = resolve_project_path(arcface_path)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 🔥 UPGRADE 1: Stricter internal thresholds & native min_face_size
        self.mtcnn = MTCNN(
            keep_all=True, 
            device=self.device, 
            post_process=False,
            min_face_size=40,            # Stop it from hallucinating tiny faces
            thresholds=[0.8, 0.9, 0.9]   # Strict P-Net, R-Net, O-Net filters
        )
        
        self.arcface = FullFaceNet(embedding_dim=512).to(self.device)
        
        # Load weights if available
        if os.path.exists(arcface_path):
            try:
                checkpoint_file = os.path.join(arcface_path, 'facenet_bulletproof_best', 'data.pkl')
                if not os.path.exists(checkpoint_file):
                    checkpoint_file = arcface_path
                
                print(f"Loading weights from {checkpoint_file}")
                checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)
                
                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint
                
                # Filter out unexpected keys like backbone.logits
                filtered_state_dict = {k: v for k, v in state_dict.items() if k in self.arcface.state_dict()}
                
                self.arcface.load_state_dict(filtered_state_dict, strict=False)
                print("Weights loaded successfully (filtered)")
            except Exception as e:
                print(f"Error loading ArcFace weights: {e}")
        
        self.arcface.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def detect_faces(self, image_path, conf_threshold=0.85, min_face_size=40):
        img = cv2.imread(image_path)
        if img is None:
            return [], None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # MTCNN detect
        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        
        faces = []
        debug_img = img.copy()
        h, w = img.shape[:2]
        
        if boxes is not None and landmarks is not None:
            for i, (box, prob, lmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob < conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                
                # Filter out tiny faces
                bw = x2 - x1
                bh = y2 - y1
                if bw < min_face_size or bh < min_face_size:
                    continue

                # 🔥 UPGRADE 2: Aspect Ratio Filter (Kills false positives)
                aspect_ratio = float(bw) / float(max(bh, 1))
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue # Not a human face shape

                # Add 20% padding
                pad_w = int(bw * 0.2)
                pad_h = int(bh * 0.2)
                
                nx1 = max(0, x1 - pad_w)
                ny1 = max(0, y1 - pad_h)
                nx2 = min(w, x2 + pad_w)
                ny2 = min(h, y2 + pad_h)
                
                face_crop = img[ny1:ny2, nx1:nx2]
                if face_crop.size > 0:
                    # Eye alignment
                    left_eye = lmark[0] # [x, y]
                    right_eye = lmark[1] # [x, y]
                    
                    aligned_face = self.align_face(face_crop, left_eye, right_eye, (nx1, ny1))
                    
                    faces.append({
                        'img': aligned_face,
                        'box': [x1, y1, x2, y2]
                    })
                    # Draw thicker red box on debug image
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        
        return faces, debug_img

    def align_face(self, face_crop, left_eye, right_eye, crop_offset):
        # Shift eyes relative to crop
        lx, ly = left_eye[0] - crop_offset[0], left_eye[1] - crop_offset[1]
        rx, ry = right_eye[0] - crop_offset[0], right_eye[1] - crop_offset[1]
        
        # Calculate angle
        dy = ry - ly
        dx = rx - lx
        
        # Prevent division by zero if MTCNN hallucinates eyes perfectly vertical
        if dx == 0:
            dx = 0.0001
            
        angle = np.degrees(np.arctan2(dy, dx))
        
        # 🔥 UPGRADE 3: Dynamic Rotation Bounds
        # Prevents cv2.warpAffine from chopping off the chin/forehead during extreme rotations
        h, w = face_crop.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to fit the rotated corners
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust the rotation matrix to take the translation into account
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Perform the rotation with the expanded boundaries
        aligned = cv2.warpAffine(face_crop, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
        
        return aligned


    def get_embedding(self, face_crop):
        # face_crop is now the aligned crop
        face_img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_tensor = self.transform(face_img_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.arcface(face_tensor)
            
        # L2 Normalization
        embedding_np = embedding.cpu().numpy().flatten()
        norm = np.linalg.norm(embedding_np)
        if norm > 0:
            embedding_np = embedding_np / norm
            
        return embedding_np


    def get_cosine_similarity(self, v1, v2):
        v1 = np.array(v1)
        v2 = np.array(v2)
        # Assuming they are already normalized, but let's be safe
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 > 0 and norm2 > 0:
            return np.dot(v1, v2) / (norm1 * norm2)
        return 0

    def cluster_faces(self, embeddings, distance_threshold=0.5, linkage='average'):
        if len(embeddings) == 0:
            return []
        
        # Agglomerative Clustering
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            linkage=linkage,
            metric='cosine'
        )
        labels = model.fit_predict(embeddings)
        return labels

