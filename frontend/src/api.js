import axios from 'axios';

const api = axios.create({
    baseURL: "http://localhost:8000"
});

export const getClusters = () => api.get('/clusters');
export const analyzeClusters = () => api.post('/analyze');
export const getAllFaces = () => api.get('/db_faces');
export const deleteFace = (faceId) => api.delete(`/db_faces/${faceId}`);
export const clearDatabase = () => api.delete('/db_faces');
export const uploadImages = (formData) => api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
});

export default api;
