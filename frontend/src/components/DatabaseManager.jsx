import React, { useState, useEffect } from 'react';
import { getAllFaces, deleteFace, clearDatabase } from '../api';

const DatabaseManager = ({ onUpdate }) => {
  const [faces, setFaces] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchFaces = async () => {
    setLoading(true);
    try {
      const response = await getAllFaces();
      setFaces(response.data);
    } catch (error) {
      console.error('Failed to fetch all faces', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFaces();
  }, []);

  const handleDelete = async (faceId) => {
    if (!window.confirm('Delete this face from database?')) return;
    try {
      await deleteFace(faceId);
      await fetchFaces();
      if (onUpdate) onUpdate();
    } catch (error) {
      console.error('Delete failed', error);
    }
  };

  const handleClearAll = async () => {
    if (!window.confirm('WARNING: This will clear the entire Qdrant database and all uploaded images. Proceed?')) return;
    try {
      await clearDatabase();
      await fetchFaces();
      if (onUpdate) onUpdate();
    } catch (error) {
      console.error('Clear failed', error);
    }
  };

  return (
    <div className="database-manager">
      <div className="db-manager-header">
        <h2>Database Inventory ({faces.length} faces)</h2>
        <div className="db-actions">
          <button className="secondary-button" onClick={fetchFaces} disabled={loading}>Refresh</button>
          <button className="clear-button" onClick={handleClearAll} disabled={loading}>Clear All</button>
        </div>
      </div>

      {loading ? (
        <div className="loading" style={{ textAlign: 'center', padding: '2rem' }}>
          <p>Syncing with Qdrant...</p>
        </div>
      ) : faces.length === 0 ? (
        <div className="no-data" style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-secondary)' }}>
          <p>Database is currently empty.</p>
        </div>
      ) : (
        <div className="faces-inventory-grid">
          {faces.map((face) => (
            <div key={face.id} className="face-inventory-item">
              <div className="face-img-container">
                <img 
                  src={`http://localhost:8000/faces/${face.payload.file_name}`} 
                  alt="Face" 
                />
              </div>
              <div className="face-info">
                <span className="face-id-label">{face.id.substring(0, 8)}</span>
                <button 
                  className="delete-small-button" 
                  onClick={() => handleDelete(face.id)}
                  title="Remove from DB"
                >
                  &times;
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default DatabaseManager;
