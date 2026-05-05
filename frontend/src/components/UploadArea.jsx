import React, { useState } from 'react';
import axios from 'axios';

const UploadArea = ({ onUploadSuccess }) => {
  const [uploading, setUploading] = useState(false);

  const handleFileChange = async (e) => {
    const files = e.target.files;
    if (!files.length) return;

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('files', files[i]);
    }

    setUploading(true);
    try {
      await axios.post('http://localhost:8000/upload', formData);
      onUploadSuccess();
    } catch (error) {
      console.error('Upload failed', error);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-section">
      <label className="upload-button" style={{ cursor: uploading ? 'wait' : 'pointer' }}>
        {uploading ? 'Processing...' : 'Upload & Detect'}
        <input 
          type="file" 
          multiple 
          accept="image/*"
          onChange={handleFileChange} 
          disabled={uploading} 
          style={{ display: 'none' }} 
        />
      </label>
    </div>
  );
};

export default UploadArea;
