import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import UploadArea from './components/UploadArea';
import ClusterGrid from './components/ClusterGrid';
import DatabaseManager from './components/DatabaseManager';

const App = () => {
  const [clusters, setClusters] = useState([]);
  const [sweepResults, setSweepResults] = useState(null);
  const [distanceThreshold, setDistanceThreshold] = useState("0.5");
  const [linkage, setLinkage] = useState("average");
  const [analyzing, setAnalyzing] = useState(false);
  const [showManager, setShowManager] = useState(false);

  const fetchClusters = async () => {
    try {
      const response = await axios.get('http://localhost:8000/clusters');
      setClusters(response.data);
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
      // Immediately set visible clusters to the default from sweep
      if (response.data.sweep_results[response.data.default_linkage]) {
        setClusters(response.data.sweep_results[response.data.default_linkage][response.data.default_threshold]);
      }
    } catch (error) {
      console.error('Analysis failed', error);
    } finally {
      setAnalyzing(false);
    }
  };

  useEffect(() => {
    if (sweepResults && sweepResults[linkage] && sweepResults[linkage][distanceThreshold]) {
      setClusters(sweepResults[linkage][distanceThreshold]);
    }
  }, [distanceThreshold, linkage, sweepResults]);

  useEffect(() => {
    fetchClusters();
  }, []);

  return (
    <div className="App">
      <nav className="navbar">
        <div className="logo">PhotoClone</div>
        <div className="nav-actions">
          {sweepResults && (
            <>
              <div className="params-container">
                <div className="param-item">
                  <label>Linkage:</label>
                  <select 
                    value={linkage} 
                    onChange={(e) => setLinkage(e.target.value)}
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
                    onChange={(e) => setDistanceThreshold(e.target.value)}
                  />
                </div>
              </div>
            </>
          )}
          <button className="secondary-button" onClick={() => setShowManager(!showManager)}>
            {showManager ? 'Hide DB Manager' : 'Manage Database'}
          </button>
          <UploadArea onUploadSuccess={fetchClusters} />
          <button className="analyze-button" onClick={handleAnalyze} disabled={analyzing}>
            {analyzing ? 'Analyzing...' : 'Analyze & Cluster'}
          </button>
        </div>
      </nav>
      
      <main className="main-content">
        <header className="page-header">
          <h1>People & Faces</h1>
          <p>Grouped by visual similarity using ArcFace and Agglomerative Clustering</p>
        </header>
        
        {showManager && <DatabaseManager onUpdate={fetchClusters} />}
        
        <ClusterGrid clusters={clusters} />
      </main>
    </div>
  );
};


export default App;
