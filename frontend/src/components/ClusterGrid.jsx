import React from 'react';

const ClusterGrid = ({ clusters }) => {
  if (!clusters || clusters.length === 0) {
    return <div className="no-clusters">No clusters detected yet. Upload photos and click "Analyze".</div>;
  }

  return (
    <div className="clusters-grid">
      {clusters.map((cluster) => (
        <div key={cluster.cluster_id} className="cluster-card">
          <h3>Person {cluster.cluster_id}</h3>
          <div className="cluster-images">
            {cluster.images.map((img, idx) => (
              <div key={idx} className="image-container">
                <img 
                  src={`http://localhost:8000/faces/${img.file_name}`} 
                  alt="Face" 
                  title={img.original_name}
                />
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ClusterGrid;
