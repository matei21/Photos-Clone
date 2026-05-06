import os



                                          



os.environ['GLOG_minloglevel'] = '2'



os.environ['MAGNOTTA_LOG_LEVEL'] = 'ERROR'







import torch



import cv2



import numpy as np



import os



from sklearn.cluster import AgglomerativeClustering



from PIL import Image



from torchvision import transforms



from ultralytics import YOLO



import mediapipe as mp



from mediapipe.tasks import python



from mediapipe.tasks.python import vision



from models_arch import FullFaceNet



from paths import resolve_project_path







class FaceProcessor:



    def __init__(self, yolo_path='models/yolov8n-face.pt', arcface_path='models/arcface.pth', landmarker_path='models/face_landmarker.task'):



        yolo_path = resolve_project_path(yolo_path)



        arcface_path = resolve_project_path(arcface_path)



        landmarker_path = resolve_project_path(landmarker_path)







                           



        self.yolo = YOLO(yolo_path)



        



                                                                 



        base_options = python.BaseOptions(model_asset_path=str(landmarker_path))



        options = vision.FaceLandmarkerOptions(



            base_options=base_options,



            output_face_blendshapes=False,



            output_facial_transformation_matrixes=False,



            num_faces=1



        )



        self.landmarker = vision.FaceLandmarker.create_from_options(options)



        



        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



        self.arcface = FullFaceNet(embedding_dim=512).to(self.device)



        



                                   



        if os.path.exists(arcface_path):



            try:



                checkpoint_file = os.path.join(arcface_path, 'facenet_bulletproof_best', 'data.pkl')



                if not os.path.exists(checkpoint_file):



                    checkpoint_file = arcface_path



                



                print(f"Loading weights from {checkpoint_file}")



                checkpoint = torch.load(checkpoint_file, map_location=self.device, weights_only=False)



                



                state_dict = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint else checkpoint



                



                                                                 



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







    def detect_faces(self, image_path, conf_threshold=0.65, min_face_size=40):



                                          



        img = cv2.imread(image_path)



        if img is None:



            return [], None



            



                                                                              



        results = self.yolo(img, verbose=False, conf=conf_threshold)



        



        faces = []



        debug_img = img.copy()



        h, w = img.shape[:2]



        



        for result in results:



            for box in result.boxes:



                                      



                if int(box.cls[0]) != 0:



                    continue







                x1, y1, x2, y2 = map(int, box.xyxy[0])



                



                                       



                bw = x2 - x1



                bh = y2 - y1



                if bw < min_face_size or bh < min_face_size:



                    continue







                                                   



                pad_w = int(bw * 0.25)



                pad_h = int(bh * 0.25)



                



                nx1 = max(0, x1 - pad_w)



                ny1 = max(0, y1 - pad_h)



                nx2 = min(w, x2 + pad_w)



                ny2 = min(h, y2 + pad_h)



                



                face_crop = img[ny1:ny2, nx1:nx2]



                if face_crop.size > 0:



                                                                    



                    aligned_face = self.align_face_local(face_crop)



                    



                    faces.append({



                        'img': aligned_face,



                        'box': [x1, y1, x2, y2]



                    })



                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 5)



        



        return faces, debug_img







    def align_face_local(self, face_crop):



                                               



        try:



                                              



            crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)



            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)



            



            detection_result = self.landmarker.detect(mp_image)



            



            if detection_result.face_landmarks:



                                                         



                landmarks = detection_result.face_landmarks[0]



                h, w = face_crop.shape[:2]



                



                                                    



                                                                    



                l_eye = landmarks[33]



                r_eye = landmarks[263]



                



                lx, ly = l_eye.x * w, l_eye.y * h



                rx, ry = r_eye.x * w, r_eye.y * h



                



                dy = ry - ly



                dx = rx - lx



                



                if dx == 0:



                    dx = 0.0001



                    



                angle = np.degrees(np.arctan2(dy, dx))



                



                                                  



                center = (w // 2, h // 2)



                M = cv2.getRotationMatrix2D(center, angle, 1.0)



                



                                                                                         



                                                                                   



                aligned = cv2.warpAffine(face_crop, M, (w, h), flags=cv2.INTER_CUBIC)



                return aligned



        except Exception as e:



            print(f"MediaPipe alignment failed: {e}")



            



        return face_crop







    def get_embedding(self, face_crop):



        face_img_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)



        face_tensor = self.transform(face_img_rgb).unsqueeze(0).to(self.device)



        



        with torch.no_grad():



            embedding = self.arcface(face_tensor)



            



                          



        embedding_np = embedding.cpu().numpy().flatten()



        norm = np.linalg.norm(embedding_np)



        if norm > 0:



            embedding_np = embedding_np / norm



            



        return embedding_np







    def get_cosine_similarity(self, v1, v2):



        v1 = np.array(v1)



        v2 = np.array(v2)



        norm1 = np.linalg.norm(v1)



        norm2 = np.linalg.norm(v2)



        if norm1 > 0 and norm2 > 0:



            return np.dot(v1, v2) / (norm1 * norm2)



        return 0







    def cluster_faces(self, embeddings, distance_threshold=0.5, linkage='average'):



        if len(embeddings) == 0:



            return []



        



        model = AgglomerativeClustering(



            n_clusters=None,



            distance_threshold=distance_threshold,



            linkage=linkage,



            metric='cosine'



        )



        labels = model.fit_predict(embeddings)



        return labels



