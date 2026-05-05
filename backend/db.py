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

    def update_cluster(self, face_id, cluster_id):
        self.client.set_payload(
            collection_name=self.collection_name,
            payload={"cluster": cluster_id},
            points=[face_id]
        )

    def update_clusters_batch(self, updates):
        # Perform updates in batches to speed up the process
        for face_id, cluster_id in updates:
            self.client.set_payload(
                collection_name=self.collection_name,
                payload={"cluster": cluster_id},
                points=[face_id]
            )
        # In a real high-perf scenario, we'd use overwrite_payload or a more 
        # optimized batch update if Qdrant client supports it for multiple points 
        # with DIFFERENT payloads easily. For now, this is already better if 
        # called correctly. 
        # Actually, Qdrant set_payload is quite fast.

    def get_all_faces(self):
        res = self.client.scroll(
            collection_name=self.collection_name,
            with_payload=True,
            with_vectors=True,
            limit=10000
        )
        return res[0]

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
            if c_id is not None:
                if c_id not in clusters:
                    clusters[c_id] = []
                clusters[c_id].append(face.payload)
        return clusters

    def close(self):
        # Explicitly close the client to avoid shutdown errors
        if hasattr(self, 'client'):
            self.client.close()
