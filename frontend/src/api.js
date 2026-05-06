import axios from 'axios';

const api = axios.create({
    baseURL: "http://localhost:8000"
});

export const getClusters = () => api.get('/clusters');
export const analyzeClusters = () => api.post('/analyze');
export const getAllFaces = () => api.get('/db_faces');
export const deleteFace = (faceId) => api.delete(`/db_faces/${faceId}`);
export const deletePhoto = (photoName) => api.delete(`/db_photos/${photoName}`);
export const clearDatabase = () => api.delete('/db_faces');
export const renameCluster = (clusterId, newName) => api.post('/rename_cluster', { cluster_id: clusterId, new_name: newName });
export const moveFace = (faceId, targetClusterId) => api.post('/move_face', { face_id: faceId, target_cluster_id: targetClusterId });
export const mergeClusters = (sourceId, targetId) => api.post('/merge_clusters', { source_cluster_id: sourceId, target_cluster_id: targetId });
export const applyClustering = (clusters) => api.post('/apply_clustering', { clusters });
export const uploadImages = (formData) => api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
});

export default api;
