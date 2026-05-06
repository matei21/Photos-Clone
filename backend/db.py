from qdrant_client import QdrantClient



from qdrant_client.http import models



import numpy as np







from paths import resolve_project_path







class QdrantDB:



    def __init__(self, path=None):



        if path is None:



            path = resolve_project_path("qdrant_db")



        self.client = QdrantClient(path=str(path))



        self.collection_name = "faces"



        self._setup_collection()







    def _setup_collection(self, force_recreate=False):



        collections = self.client.get_collections().collections



        exists = any(c.name == self.collection_name for c in collections)



        



        if force_recreate or not exists:



            self.client.recreate_collection(



                collection_name=self.collection_name,



                vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE),



            )







    def add_face(self, face_id, embedding, metadata):



        self.client.upsert(



            collection_name=self.collection_name,



            points=[



                models.PointStruct(



                    id=face_id,



                    vector=embedding.tolist(),



                    payload=metadata



                )



            ]



        )







    def add_faces_batch(self, faces):



        if not faces:



            return







        points = [



            models.PointStruct(



                id=face_id,



                vector=embedding.tolist(),



                payload=metadata



            )



            for face_id, embedding, metadata in faces



        ]







        self.client.upsert(



            collection_name=self.collection_name,



            points=points



        )







    def update_cluster(self, face_id, cluster_id):



        self.client.set_payload(



            collection_name=self.collection_name,



            payload={"cluster": cluster_id},



            points=[face_id]



        )







    def update_clusters_batch(self, updates):



        if not updates:



            return



            



                                                



        cluster_groups = {}



        for face_id, cluster_id in updates:



            cluster_groups.setdefault(cluster_id, []).append(face_id)



            



                                                                         



        for c_id, face_ids in cluster_groups.items():



            self.client.set_payload(



                collection_name=self.collection_name,



                payload={"cluster": c_id, "cluster_name": None},



                points=face_ids



            )



        print(f"Successfully batch-updated payloads for {len(updates)} faces across {len(cluster_groups)} clusters.")







    def delete_photo_faces(self, debug_full_image_name):



        """Delete all face entries associated with a specific original photo"""



        faces = self.get_all_faces()



        points_to_delete = [f.id for f in faces if f.payload.get("debug_full_image") == debug_full_image_name]



        



        if points_to_delete:



            self.client.delete(



                collection_name=self.collection_name,



                points_selector=models.PointIdsList(



                    points=points_to_delete



                )



            )



        return points_to_delete







    def get_all_faces(self):



        try:



            res = self.client.scroll(



                collection_name=self.collection_name,



                with_payload=True,



                with_vectors=True,



                limit=10000



            )



            return res[0]



        except IndexError:



                                                                                     



            return []



        except Exception as e:



            print(f"Error fetching faces from Qdrant: {e}")



            return []







    def delete_face(self, face_id):



        self.client.delete(



            collection_name=self.collection_name,



            points_selector=models.PointIdsList(



                points=[face_id]



            )



        )







    def clear_all(self):



        self._setup_collection(force_recreate=True)







    def get_clusters(self):



        faces = self.get_all_faces()



        clusters = {}



        for face in faces:



            c_id = face.payload.get("cluster")



            if c_id is None:



                c_id = -1                                   



                



            if c_id not in clusters:



                clusters[c_id] = []



            



                                                                  



            payload = face.payload.copy()



            payload["id"] = face.id



            clusters[c_id].append(payload)



        return clusters







    def update_cluster_name(self, cluster_id, name):



                                                   



        faces = self.get_all_faces()



        points_to_update = [f.id for f in faces if f.payload.get("cluster") == cluster_id]



        



        if points_to_update:



            self.client.set_payload(



                collection_name=self.collection_name,



                payload={"cluster_name": name},



                points=points_to_update



            )







    def move_face_to_cluster(self, face_id, target_cluster_id, target_cluster_name=None):



        payload = {"cluster": target_cluster_id}



        if target_cluster_name:



            payload["cluster_name"] = target_cluster_name



        else:



                                                                  



            faces = self.get_all_faces()



            existing_name = next((f.payload.get("cluster_name") for f in faces if f.payload.get("cluster") == target_cluster_id), None)



            if existing_name:



                payload["cluster_name"] = existing_name







        self.client.set_payload(



            collection_name=self.collection_name,



            payload=payload,



            points=[face_id]



        )







    def close(self):



                                                              



        if hasattr(self, 'client'):



            self.client.close()







    def merge_clusters(self, source_cluster_id, target_cluster_id):



                                             



        faces = self.get_all_faces()



        source_points = [f.id for f in faces if f.payload.get("cluster") == source_cluster_id]



        



        if not source_points:



            return



            



                                                  



        target_name = next((f.payload.get("cluster_name") for f in faces if f.payload.get("cluster") == target_cluster_id), None)



        



                                                                



        payload = {"cluster": target_cluster_id}



        if target_name:



            payload["cluster_name"] = target_name



            



        self.client.set_payload(



            collection_name=self.collection_name,



            payload=payload,



            points=source_points



        )



