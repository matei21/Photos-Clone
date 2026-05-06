import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import UploadArea from './components/UploadArea';
import ClusterGrid from './components/ClusterGrid';
import DatabaseManager from './components/DatabaseManager';
import { moveFace, renameCluster, applyClustering, mergeClusters, deleteFace, deletePhoto } from './api';

const App = () => {
  const [clusters, setClusters] = useState([]);
  const [sweepResults, setSweepResults] = useState(null);
  const [distanceThreshold, setDistanceThreshold] = useState("0.5");
  const [linkage, setLinkage] = useState("average");
  const [analyzing, setAnalyzing] = useState(false);
  const [showManager, setShowManager] = useState(false);
  const [viewMode, setViewMode] = useState('grid'); 
  const [selectedClusterId, setSelectedClusterId] = useState(null);
  const [isPreview, setIsPreview] = useState(false);
  const [isMerging, setIsMerging] = useState(false);
  const [mergeTargetId, setMergeTargetId] = useState("");
  const scrollContainerRef = React.useRef(null);

  const fetchClusters = async () => {
    try {
      const response = await axios.get('http://localhost:8000/clusters');
      setClusters(response.data);
      setIsPreview(false);
      if (response.data.length > 0 && selectedClusterId === null) {
        setSelectedClusterId(response.data[0].cluster_id);
      }
    } catch (error) {
      console.error('Failed to fetch clusters', error);
    }
  };

  const handleAnalyze = async () => {
    setAnalyzing(true);
    try {
      const response = await axios.post('http://localhost:8000/analyze');
      setSweepResults(response.data.sweep_results);
      setDistanceThreshold(response.data.default_threshold);
      setLinkage(response.data.default_linkage);
      setIsPreview(true);
      
      const defaultClusters = response.data.sweep_results[response.data.default_linkage][response.data.default_threshold];
      setClusters(defaultClusters);
      if (defaultClusters.length > 0) {
        setSelectedClusterId(defaultClusters[0].cluster_id);
      }
    } catch (error) {
      console.error('Analysis failed', error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleApplyClustering = async () => {
    try {
      console.log("Applying clustering for", clusters.length, "clusters");
      await applyClustering(clusters);
      setSweepResults(null);
      setIsPreview(false); 
      await fetchClusters();
      alert("Clustering applied to database!");
    } catch (error) {
      console.error("Failed to apply clustering", error);
      alert("Failed to apply clustering. See console.");
    }
  };

  const handleMoveFace = async (faceId, targetClusterId) => {
    try {
      await moveFace(faceId, targetClusterId);
      if (!isPreview) {
        await fetchClusters();
      } else {
        
        alert("Face moved in DB. Note: You are currently in preview mode.");
      }
    } catch (error) {
      console.error('Failed to move face', error);
    }
  };

  const handleRenameCluster = async (clusterId, newName) => {
    
    setClusters(prev => prev.map(c => 
      c.cluster_id === clusterId ? { ...c, cluster_name: newName } : c
    ));

    try {
      await renameCluster(clusterId, newName);
    } catch (error) {
      console.error('Failed to rename cluster', error);
    }
  };

  const handleOnDeleteFace = async (faceId) => {
    try {
      await deleteFace(faceId);
      await fetchClusters();
    } catch (error) {
      console.error("Failed to delete face", error);
    }
  };

  const handleOnDeletePhoto = async (photoName) => {
    if (!window.confirm("This will delete the original photo and ALL detections in it. Proceed?")) return;
    try {
      await deletePhoto(photoName);
      await fetchClusters();
    } catch (error) {
      console.error("Failed to delete photo", error);
    }
  };

  const handleMergeClusters = async () => {
    if (!selectedClusterId || !mergeTargetId) return;
    if (selectedClusterId === parseInt(mergeTargetId)) {
        alert("Cannot merge a cluster into itself.");
        return;
    }

    try {
      await mergeClusters(selectedClusterId, mergeTargetId);
      setIsMerging(false);
      setMergeTargetId("");
      await fetchClusters();
    } catch (error) {
      console.error('Failed to merge clusters', error);
    }
  };

  const scrollSelector = (direction) => {
    if (scrollContainerRef.current) {
      const scrollAmount = direction === 'left' ? -300 : 300;
      scrollContainerRef.current.scrollBy({ left: scrollAmount, behavior: 'smooth' });
    }
  };

  useEffect(() => {
    if (sweepResults && sweepResults[linkage] && sweepResults[linkage][distanceThreshold]) {
      const currentClusters = sweepResults[linkage][distanceThreshold];
      setClusters(currentClusters);
      setIsPreview(true);
      
      
      if (selectedClusterId !== null) {
        const exists = currentClusters.some(c => c.cluster_id === selectedClusterId);
        if (!exists && currentClusters.length > 0) {
          setSelectedClusterId(currentClusters[0].cluster_id);
        }
      } else if (currentClusters.length > 0) {
        setSelectedClusterId(currentClusters[0].cluster_id);
      }
    }
  }, [distanceThreshold, linkage, sweepResults]);

  useEffect(() => {
    fetchClusters();
  }, []);

  const selectedCluster = clusters.find(c => c.cluster_id === selectedClusterId);

  
  const deduplicatedImages = React.useMemo(() => {
    if (!selectedCluster) return [];
    const seen = new Set();
    return selectedCluster.images.filter(img => {
      if (seen.has(img.original_name)) return false;
      seen.add(img.original_name);
      return true;
    });
  }, [selectedCluster]);

  return (
    <div className="App">
      <nav className="navbar">
        <div className="logo">
          <span>SmileCluster</span>
        </div>
        <div className="nav-actions">
          {isPreview && (
            <button className="primary-button" onClick={handleApplyClustering} style={{ backgroundColor: '#28a745' }}>
              Apply & Save Clusters
            </button>
          )}
          
          <button 
            className="secondary-button" 
            onClick={() => setViewMode(viewMode === 'grid' ? 'timeline' : 'grid')}
          >
            {viewMode === 'grid' ? 'Memory Timeline' : 'Debug Grid'}
          </button>
          
          {sweepResults && viewMode === 'grid' && (
            <div className="params-container">
              <div className="param-item">
                <label>Linkage</label>
                <select 
                  value={linkage} 
                  onChange={(e) => {
                    setLinkage(e.target.value);
                    setIsPreview(true);
                  }}
                  className="linkage-select"
                >
                  <option value="average">Average</option>
                  <option value="complete">Complete</option>
                </select>
              </div>
              <div className="param-item">
                <label>Distance: {distanceThreshold}</label>
                <input 
                  type="range" 
                  min="0.10" 
                  max="0.90" 
                  step="0.01" 
                  value={distanceThreshold} 
                  onChange={(e) => {
                    setDistanceThreshold(e.target.value);
                    setIsPreview(true);
                  }}
                />
              </div>
            </div>
          )}
          
          <button className="secondary-button" onClick={() => setShowManager(!showManager)}>
            {showManager ? 'Hide DB' : 'Manage DB'}
          </button>
          <UploadArea onUploadSuccess={fetchClusters} />
          <button className="analyze-button" onClick={handleAnalyze} disabled={analyzing}>
            {analyzing ? 'Analyzing...' : 'Cluster Now'}
          </button>
        </div>
      </nav>
      
      <main className="main-content">
        <header className="page-header">
          <h1>
            {viewMode === 'grid' ? 'Faces Inventory' : 'Memory Timeline'}
            {isPreview && <span style={{ fontSize: '0.9rem', color: '#e67e22', marginLeft: '1rem' }}>(Preview Mode)</span>}
          </h1>
          <p>
            {viewMode === 'grid' 
              ? 'Deep dive into the detected faces and clustering parameters.'
              : 'Relive moments captured for each unique person.'}
          </p>
        </header>
        
        {showManager && (
          <div className="manager-wrapper">
            <DatabaseManager onUpdate={fetchClusters} />
          </div>
        )}
        
        <div className="view-container">
          {viewMode === 'grid' ? (
            <ClusterGrid 
                clusters={clusters} 
                onMoveFace={handleMoveFace} 
                isPreview={isPreview} 
                onDeleteFace={handleOnDeleteFace}
            />
          ) : (
            <div className="timeline-view">
              {clusters.length > 0 ? (
                <>
                  <div className="person-selector-wrapper">
                    <button className="scroll-button" onClick={() => scrollSelector('left')}>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M15 18l-6-6 6-6"/></svg>
                    </button>
                    <div className="person-selector" ref={scrollContainerRef}>
                      {clusters.map((cluster) => (
                        <div 
                          key={cluster.cluster_id} 
                          className={`selector-item ${selectedClusterId === cluster.cluster_id ? 'active' : ''}`}
                          onClick={() => setSelectedClusterId(cluster.cluster_id)}
                        >
                          <img src={`http://localhost:8000/faces/${cluster.images[0].file_name}`} alt="Face" />
                          <span>{cluster.cluster_name || `Person ${cluster.cluster_id}`}</span>
                        </div>
                      ))}
                    </div>
                    <button className="scroll-button" onClick={() => scrollSelector('right')}>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M9 18l6-6-6-6"/></svg>
                    </button>
                  </div>

                  {selectedCluster && (
                    <div className="focused-person-story">
                      <div className="person-header">
                        <div className="representative-face">
                          <img src={`http://localhost:8000/faces/${selectedCluster.images[0].file_name}`} alt="Representative" />
                        </div>
                        <div className="person-meta">
                          <div className="title-row" style={{ display: 'flex', alignItems: 'center' }}>
                            <input 
                                key={`name-${selectedCluster.cluster_id}`}
                                className="editable-title"
                                defaultValue={selectedCluster.cluster_name || `Person ${selectedCluster.cluster_id}`}
                                onBlur={(e) => {
                                if (e.target.value !== (selectedCluster.cluster_name || `Person ${selectedCluster.cluster_id}`)) {
                                    handleRenameCluster(selectedCluster.cluster_id, e.target.value);
                                }
                                }}
                                onKeyDown={(e) => e.key === 'Enter' && e.target.blur()}
                                title="Click to rename"
                            />
                            {!isPreview && (
                                <button 
                                    className="secondary-button merge-toggle" 
                                    onClick={() => setIsMerging(!isMerging)}
                                    style={{ marginLeft: '1rem', padding: '0.4rem 0.8rem', fontSize: '0.75rem' }}
                                >
                                    {isMerging ? 'Cancel Merge' : 'Merge with...'}
                                </button>
                            )}
                          </div>
                          
                          {isMerging && (
                            <div className="merge-panel" style={{ marginTop: '0.5rem', display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                                <select 
                                    value={mergeTargetId} 
                                    onChange={(e) => setMergeTargetId(e.target.value)}
                                    className="reassignment-select"
                                    style={{ fontSize: '0.8rem', padding: '0.4rem' }}
                                >
                                    <option value="">Select Person...</option>
                                    {clusters
                                        .filter(c => c.cluster_id !== selectedClusterId && c.cluster_id !== -1)
                                        .map(c => (
                                            <option key={c.cluster_id} value={c.cluster_id}>
                                                {c.cluster_name || `Person ${c.cluster_id}`}
                                            </option>
                                        ))
                                    }
                                </select>
                                <button 
                                    className="primary-button" 
                                    onClick={handleMergeClusters}
                                    disabled={!mergeTargetId}
                                    style={{ padding: '0.4rem 0.8rem', fontSize: '0.75rem' }}
                                >
                                    Confirm Merge
                                </button>
                            </div>
                          )}
                          <span className="photo-count" style={{ display: 'block', marginTop: '0.5rem' }}>{deduplicatedImages.length} memories found</span>
                        </div>
                      </div>
                      <div className="full-photos-grid">
                        {deduplicatedImages.map((img, idx) => (
                          <div key={idx} className="full-photo-card">
                            <div className="photo-display-container" style={{ position: 'relative' }}>
                                {img.debug_full_image ? (
                                    <>
                                        <img src={`http://localhost:8000/debug_faces/${img.debug_full_image}`} alt={img.original_name} />
                                        <button 
                                            className="delete-photo-button"
                                            onClick={() => handleOnDeletePhoto(img.debug_full_image)}
                                            title="Delete entire photo"
                                        >
                                            &times;
                                        </button>
                                    </>
                                ) : (
                                    <div className="no-full-image">Original Image Not Available</div>
                                )}
                            </div>
                            <span className="photo-label">{img.original_name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="no-clusters">
                  <h3>No memories found yet</h3>
                  <p>Upload some photos and run the analysis to see them here.</p>
                </div>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
};




export default App;
