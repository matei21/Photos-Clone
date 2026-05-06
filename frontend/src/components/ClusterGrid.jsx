import React, { useState } from 'react';

const ClusterGrid = ({ clusters, onMoveFace, isPreview, onDeleteFace }) => {
  const [selectedFace, setSelectedFace] = useState(null);
  const [targetCluster, setTargetCluster] = useState("");

  if (!clusters || clusters.length === 0) {
    return (
      <div className="no-clusters">
        <h3>No faces detected yet</h3>
        <p>Upload photos and click "Cluster Now" to see results here.</p>
      </div>
    );
  }

  const handleFaceClick = (img) => {
    if (isPreview) return;
    if (selectedFace && selectedFace.file_name === img.file_name) {
      setSelectedFace(null);
    } else {
      setSelectedFace(img);
    }
  };

  const handleMove = async () => {
    if (!selectedFace || !targetCluster) return;
    
    
    const faceId = selectedFace.id || selectedFace.file_name.substring(0, selectedFace.file_name.lastIndexOf('.'));
    
    console.log(`Requesting move for face ${faceId} to cluster ${targetCluster}`);
    
    try {
      await onMoveFace(faceId, targetCluster);
      setSelectedFace(null);
      setTargetCluster("");
    } catch (err) {
      console.error("Move failed in component", err);
    }
  };

  return (
    <div className="grid-view-wrapper">
      {selectedFace && !isPreview && (
        <div className="reassignment-panel">
          <p>Reassign selected face to:</p>
          <div className="reassignment-actions">
            <select 
              className="reassignment-select"
              value={targetCluster}
              onChange={(e) => setTargetCluster(e.target.value)}
            >
              <option value="">Select Cluster...</option>
              <option value="new">+ Create New Cluster</option>
              {clusters.map(c => (
                <option key={c.cluster_id} value={c.cluster_id}>
                  {c.cluster_name || `Person ${c.cluster_id}`}
                </option>
              ))}
            </select>
            <button 
              className="primary-button" 
              onClick={handleMove}
              disabled={!targetCluster}
            >
              Switch Cluster
            </button>
            <button className="secondary-button" onClick={() => setSelectedFace(null)}>Cancel</button>
          </div>
        </div>
      )}

      <div className="clusters-grid">
        {clusters.map((cluster) => (
          <div key={cluster.cluster_id} className="cluster-card">
            <h3>
              <span>{cluster.cluster_name || `Person ${cluster.cluster_id}`}</span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
                {cluster.images.length} {cluster.images.length === 1 ? 'face' : 'faces'}
              </span>
            </h3>
            <div className="cluster-images">
              {cluster.images.map((img, idx) => (
                <div 
                  key={idx} 
                  className={`image-container ${selectedFace?.file_name === img.file_name ? 'selected' : ''}`}
                  onClick={() => handleFaceClick(img)}
                >
                  <img 
                    src={`http://localhost:8000/faces/${img.file_name}`} 
                    alt="Face" 
                    title={img.original_name}
                  />
                  {!isPreview && (
                    <button 
                      className="delete-overlay-button"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm("Delete this detection?")) {
                            onDeleteFace(img.id || img.file_name.substring(0, img.file_name.lastIndexOf('.')));
                        }
                      }}
                      title="Delete detection"
                    >
                      &times;
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ClusterGrid;
