1. The False Positives (Background Faces)
Right now, you are running MTCNN on its default internal thresholds, which are quite loose ([0.6, 0.7, 0.7]). It acts like an overeager puppy, seeing faces in clouds and curtains. We need to raise its internal strictness and add a mathematical Aspect Ratio Filter (human faces are generally square-ish; a box that is 3 times taller than it is wide is a false positive).

2. The False Negatives (Missed Faces)
MTCNN is a cascade network. By default, it searches for faces as small as 20x20 pixels. When it wastes computing power looking for tiny faces, it sometimes hallucinates or misses the obvious big ones. We need to tell the network's constructor to stop looking for microscopic faces.

3. The Closed-Eyes Rotation Bug
When eyes are closed, MTCNN struggles to pinpoint the exact pupil. Sometimes it hallucinates the "left" eye being perfectly level with the "right" eye, resulting in an angle of 0.0 (which leaves the image sideways).
Furthermore, there is a hidden bug in your align_face function: cv2.warpAffine(..., (w, h)) rotates the image inside its original box. If a face is rotated 90 degrees, this math actually chops off the chin and forehead! We need to dynamically resize the box during rotation so no pixels are lost.

The Code Fixes
Replace your __init__, detect_faces, and align_face methods with these upgraded versions.

1. The Updated __init__
Python
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
        
        # ... (Keep your exact weight loading logic here, it is perfectly fine) ...
        
        self.arcface.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
2. The Updated detect_faces (with Aspect Ratio filtering)
Python
    def detect_faces(self, image_path, conf_threshold=0.85, min_face_size=40):
        img = cv2.imread(image_path)
        if img is None:
            return [], None
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        boxes, probs, landmarks = self.mtcnn.detect(pil_img, landmarks=True)
        
        faces = []
        debug_img = img.copy()
        h, w = img.shape[:2]
        
        if boxes is not None and landmarks is not None:
            for i, (box, prob, lmark) in enumerate(zip(boxes, probs, landmarks)):
                if prob < conf_threshold:
                    continue
                
                x1, y1, x2, y2 = map(int, box)
                bw = x2 - x1
                bh = y2 - y1
                
                # Filter out tiny faces
                if bw < min_face_size or bh < min_face_size:
                    continue

                # 🔥 UPGRADE 2: Aspect Ratio Filter (Kills false positives)
                aspect_ratio = float(bw) / float(max(bh, 1))
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:
                    continue # Not a human face shape

                # Add padding
                pad_w = int(bw * 0.2)
                pad_h = int(bh * 0.2)
                
                nx1 = max(0, x1 - pad_w)
                ny1 = max(0, y1 - pad_h)
                nx2 = min(w, x2 + pad_w)
                ny2 = min(h, y2 + pad_h)
                
                face_crop = img[ny1:ny2, nx1:nx2]
                
                if face_crop.size > 0:
                    # Eye alignment
                    left_eye = lmark[0] 
                    right_eye = lmark[1] 
                    
                    aligned_face = self.align_face(face_crop, left_eye, right_eye, (nx1, ny1))
                    
                    faces.append({
                        'img': aligned_face,
                        'box': [x1, y1, x2, y2]
                    })
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 5)
        
        return faces, debug_img
3. The Updated align_face (Dynamic Cropping)
Python
    def align_face(self, face_crop, left_eye, right_eye, crop_offset):
        # Shift eyes relative to crop
        lx, ly = left_eye[0] - crop_offset[0], left_eye[1] - crop_offset[1]
        rx, ry = right_eye[0] - crop_offset[0], right_eye[1] - crop_offset[1]
        
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