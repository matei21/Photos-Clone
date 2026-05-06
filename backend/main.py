import os



import sys







                                           



os.environ['GLOG_minloglevel'] = '3'



os.environ['MAGNOTTA_LOG_LEVEL'] = 'ERROR'



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'







                                                                              



                                                         



                         



                                    







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



                                                                       



                                            



    yield



                    



    db.close()







app = FastAPI(lifespan=lifespan)







                                                  



app.add_middleware(



    CORSMiddleware,



    allow_origins=["*"],



    allow_credentials=True,



    allow_methods=["*"],



    allow_headers=["*"],



)







       



UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "faces")



DEBUG_DIR = os.path.join(os.path.dirname(__file__), "debug_faces")



os.makedirs(UPLOAD_DIR, exist_ok=True)



os.makedirs(DEBUG_DIR, exist_ok=True)







                             



                                                            



try:



    db = QdrantDB()



    processor = FaceProcessor()



    print("Backend Services Initialized Successfully")



except Exception as e:



    print(f"CRITICAL: Failed to initialize backend services: {e}")







@app.post("/upload")



async def upload_images(files: List[UploadFile] = File(...)):



    print(f"Received upload request with {len(files)} files")



    success_count = 0



    batch_faces = []



    for file in files:



        if not file.filename:



            continue



        



        print(f"Processing file: {file.filename}")



                                                                   



        temp_id = str(uuid.uuid4())



        file_extension = os.path.splitext(file.filename)[1]



        if not file_extension:



            file_extension = ".jpg"



            



        temp_name = f"temp_{temp_id}{file_extension}"



        temp_path = os.path.join(UPLOAD_DIR, temp_name)



        



        try:



            with open(temp_path, "wb") as buffer:



                shutil.copyfileobj(file.file, buffer)



        except Exception as e:



            print(f"Failed to save temporary file {file.filename}: {e}")



            continue



        



                         



        try:



            faces, debug_img = processor.detect_faces(temp_path)



        except Exception as e:



            print(f"Error during face detection for {file.filename}: {e}")



            faces = []



        



                                      



        debug_name = f"full_{temp_id}{file_extension}"



        debug_path = os.path.join(DEBUG_DIR, debug_name)



        shutil.copy(temp_path, debug_path)



        



                                       



        for face_data in faces:



            try:



                face_id = str(uuid.uuid4())



                embedding = processor.get_embedding(face_data['img'])



                



                crop_name = f"{face_id}{file_extension}"



                crop_path = os.path.join(UPLOAD_DIR, crop_name)



                cv2.imwrite(crop_path, face_data['img'])



                



                clean_box = [int(x) for x in face_data['box']]







                batch_faces.append((



                    face_id,



                    embedding,



                    {



                        "file_name": crop_name,



                        "original_name": file.filename,



                        "debug_full_image": debug_name,



                        "cluster": None,



                        "box": clean_box



                    }



                ))



            except Exception as e:



                print(f"Failed to process face in {file.filename}: {e}")



        



                 



        if os.path.exists(temp_path):



            os.remove(temp_path)



        success_count += 1







    if batch_faces:



        try:



            db.add_faces_batch(batch_faces)



            print(f"Batch inserted {len(batch_faces)} faces into Qdrant")



        except Exception as e:



            print(f"Failed batch insert into Qdrant: {e}")



    



    return {"message": f"Successfully uploaded {success_count} files"}







@app.post("/analyze")



async def analyze_clusters():



    faces = db.get_all_faces()



    if not faces:



        return {"message": "No faces found to analyze"}



    



    embeddings = [face.vector for face in faces]



    ids = [face.id for face in faces]



    



                                                                        



    threshold_values = [round(x * 0.01, 2) for x in range(10, 91)]



    linkages = ['average', 'complete']



    sweep_data = {linkage: {} for linkage in linkages}



    



    for linkage in linkages:



        for threshold in threshold_values:



            labels = processor.cluster_faces(embeddings, distance_threshold=threshold, linkage=linkage)



            



                                                                     



            clusters = {}



            for face_id, label in zip(ids, labels):



                                                                 



                c_id = int(label)



                



                if c_id not in clusters:



                    clusters[c_id] = []



                



                                                             



                face_meta = next(f.payload for f in faces if f.id == face_id).copy()



                face_meta["id"] = face_id                             



                clusters[c_id].append(face_meta)



                



                                                 



            sweep_data[linkage][str(threshold)] = [



                {"cluster_id": c_id, "images": imgs} 



                for c_id, imgs in clusters.items()



            ]



        



    return {



        "sweep_results": sweep_data,



        "default_threshold": "0.5",



        "default_linkage": "average"



    }







@app.post("/apply_clustering")



async def apply_clustering(data: dict):



    clusters = data.get("clusters")                                            



    if not clusters:



        print("Apply clustering failed: No clusters provided in data")



        return {"message": "No clusters provided"}, 400



    



    updates = []



    for cluster in clusters:



        c_id = cluster.get("cluster_id")



        for img in cluster.get("images", []):



            face_id = img.get("id")



            if face_id:



                updates.append((face_id, int(c_id)))



    



    print(f"Applying clustering: {len(updates)} faces across {len(clusters)} clusters")



    



    if updates:



        try:



            db.update_clusters_batch(updates)



            print("Successfully applied clustering to Qdrant")



            return {"message": f"Successfully applied {len(updates)} cluster assignments"}



        except Exception as e:



            print(f"Error applying clusters: {e}")



            return {"message": str(e)}, 500



    print("No face IDs found to update")



    return {"message": "No updates to perform"}







@app.get("/clusters")



async def get_clusters():



    clusters = db.get_clusters()



                         



    result = []



    for c_id, faces in clusters.items():



                                                    



        c_name = None



        for f in faces:



            if f.get("cluster_name"):



                c_name = f.get("cluster_name")



                break



        



        if c_id == -1:



            c_name = "New Detections (Unclustered)"



        



        result.append({



            "cluster_id": c_id,



            "cluster_name": c_name,



            "images": faces



        })



    return result







@app.post("/rename_cluster")



async def rename_cluster(data: dict):



    cluster_id = data.get("cluster_id")



    new_name = data.get("new_name")



    print(f"Renaming cluster {cluster_id} to {new_name}")



    if cluster_id is not None and new_name:



        try:



            db.update_cluster_name(int(cluster_id), new_name)



            return {"message": f"Cluster {cluster_id} renamed to {new_name}"}



        except Exception as e:



            print(f"Error renaming cluster: {e}")



            return {"message": str(e)}, 500



    return {"message": "Invalid data"}, 400







@app.post("/move_face")



async def move_face(data: dict):



    face_id = data.get("face_id")



    target_cluster_id = data.get("target_cluster_id")



    print(f"Moving face {face_id} to cluster {target_cluster_id}")



    



                                                             



    if target_cluster_id == "new":



        clusters = db.get_clusters()



        if not clusters:



            target_cluster_id = 0



        else:



            target_cluster_id = max(clusters.keys()) + 1



            



    if face_id and target_cluster_id is not None:



        try:



            db.move_face_to_cluster(face_id, int(target_cluster_id))



            return {"message": f"Face {face_id} moved to cluster {target_cluster_id}"}



        except Exception as e:



            print(f"Error moving face: {e}")



            return {"message": str(e)}, 500



    return {"message": "Invalid data"}, 400







@app.get("/db_faces")



async def get_all_faces():



    faces = db.get_all_faces()



    return [{"id": f.id, "payload": f.payload} for f in faces]







@app.delete("/db_faces/{face_id}")



async def delete_face(face_id: str):



                                                                



    all_faces = db.get_all_faces()



    target_face = next((f for f in all_faces if f.id == face_id), None)



    



    db.delete_face(face_id)



    



    if target_face:



        file_name = target_face.payload.get("file_name")



                                                



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







@app.post("/merge_clusters")



async def merge_clusters(data: dict):



    source_id = data.get("source_cluster_id")



    target_id = data.get("target_cluster_id")



    



    if source_id is not None and target_id is not None:



        try:



            db.merge_clusters(int(source_id), int(target_id))



            return {"message": f"Cluster {source_id} merged into {target_id}"}



        except Exception as e:



            print(f"Error merging clusters: {e}")



            return {"message": str(e)}, 500



    return {"message": "Invalid data"}, 400







@app.delete("/db_photos/{photo_name}")



async def delete_original_photo(photo_name: str):



                                                            



    deleted_ids = db.delete_photo_faces(photo_name)



    



                                                   



    file_path = os.path.join(DEBUG_DIR, photo_name)



    if os.path.exists(file_path):



        os.remove(file_path)



        



                                                                 



                                                               



                                             



    



    return {"message": f"Photo {photo_name} and its {len(deleted_ids)} detections deleted"}







app.mount("/faces", StaticFiles(directory=UPLOAD_DIR), name="faces")



app.mount("/debug_faces", StaticFiles(directory=DEBUG_DIR), name="debug_faces")







if __name__ == "__main__":



    uvicorn.run(app, host="0.0.0.0", port=8000)



