"""
File Handlers
Specialized handlers for different file formats.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import os
import json
import pickle
from PIL import Image
import h5py


class BaseFileHandler:
    """Base class for file handlers."""
    
    @staticmethod
    def save(filepath: str, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save data to file."""
        raise NotImplementedError
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data from file."""
        raise NotImplementedError


class CSVHandler(BaseFileHandler):
    """CSV file handler for tabular data."""
    
    @staticmethod
    def save(filepath: str, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save data to CSV file."""
        if data.ndim == 1:
            df = pd.DataFrame({'value': data})
        elif data.ndim == 2:
            df = pd.DataFrame(data)
        else:
            # Flatten higher dimensional data
            flattened = data.reshape(data.shape[0], -1)
            df = pd.DataFrame(flattened)
        
        df.to_csv(filepath, index=False)
        
        # Save metadata to separate JSON file
        if metadata:
            metadata_path = filepath.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data from CSV file."""
        df = pd.read_csv(filepath)
        data = df.values
        
        # Load metadata if available
        metadata_path = filepath.replace('.csv', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        metadata['columns'] = list(df.columns)
        metadata['shape'] = data.shape
        
        return data, metadata


class NumpyHandler(BaseFileHandler):
    """NumPy file handler for efficient binary storage."""
    
    @staticmethod
    def save(filepath: str, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save data to NumPy binary file."""
        np.save(filepath, data)
        
        # Save metadata to separate JSON file
        if metadata:
            metadata_path = filepath.replace('.npy', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data from NumPy binary file."""
        data = np.load(filepath)
        
        # Load metadata if available
        metadata_path = filepath.replace('.npy', '_metadata.json')
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        metadata['shape'] = data.shape
        metadata['dtype'] = str(data.dtype)
        
        return data, metadata


class NPZHandler(BaseFileHandler):
    """Compressed NumPy file handler with embedded metadata."""
    
    @staticmethod
    def save(filepath: str, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save data to compressed NumPy file."""
        save_dict = {'data': data}
        if metadata:
            save_dict['metadata'] = json.dumps(metadata)
        
        np.savez_compressed(filepath, **save_dict)
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data from compressed NumPy file."""
        loaded = np.load(filepath)
        
        if 'data' in loaded:
            data = loaded['data']
        else:
            # Try to get the first array
            keys = list(loaded.keys())
            data = loaded[keys[0]]
        
        # Extract metadata
        metadata = {}
        if 'metadata' in loaded:
            metadata = json.loads(loaded['metadata'].item())
        
        return data, metadata


class ImageHandler(BaseFileHandler):
    """Image file handler for visualization and export."""
    
    @staticmethod
    def save(filepath: str, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save data as image file."""
        # Normalize data to 0-255 range
        if data.dtype != np.uint8:
            data_norm = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)
        else:
            data_norm = data
        
        # Handle different data shapes
        if data_norm.ndim == 2:
            # Grayscale image
            img = Image.fromarray(data_norm, mode='L')
        elif data_norm.ndim == 3 and data_norm.shape[2] in [3, 4]:
            # RGB or RGBA image
            img = Image.fromarray(data_norm)
        else:
            raise ValueError(f"Unsupported image shape: {data_norm.shape}")
        
        img.save(filepath)
        
        # Save metadata to separate JSON file
        if metadata:
            metadata_path = filepath.rsplit('.', 1)[0] + '_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data from image file."""
        img = Image.open(filepath)
        data = np.array(img)
        
        # Load metadata if available
        metadata_path = filepath.rsplit('.', 1)[0] + '_metadata.json'
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        metadata['shape'] = data.shape
        metadata['dtype'] = str(data.dtype)
        metadata['mode'] = img.mode
        
        return data, metadata


class HDF5Handler(BaseFileHandler):
    """HDF5 file handler for large datasets and hierarchical storage."""
    
    @staticmethod
    def save(filepath: str, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save data to HDF5 file."""
        with h5py.File(filepath, 'w') as f:
            # Create dataset
            dataset = f.create_dataset('data', data=data)
            
            # Save metadata as attributes
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float)):
                        dataset.attrs[key] = value
                    else:
                        # Convert complex objects to JSON string
                        dataset.attrs[key] = json.dumps(value)
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data from HDF5 file."""
        with h5py.File(filepath, 'r') as f:
            data = f['data'][:]
            
            # Extract metadata from attributes
            metadata = {}
            for key, value in f['data'].attrs.items():
                try:
                    # Try to parse as JSON first
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    metadata[key] = value
        
        return data, metadata


class PickleHandler(BaseFileHandler):
    """Pickle file handler for Python objects."""
    
    @staticmethod
    def save(filepath: str, data: np.ndarray, metadata: Optional[Dict[str, Any]] = None):
        """Save data using pickle."""
        save_dict = {'data': data, 'metadata': metadata or {}}
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @staticmethod
    def load(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data using pickle."""
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        
        data = loaded['data']
        metadata = loaded.get('metadata', {})
        
        return data, metadata


class FileHandlerFactory:
    """Factory class for creating appropriate file handlers."""
    
    _handlers = {
        '.csv': CSVHandler,
        '.npy': NumpyHandler,
        '.npz': NPZHandler,
        '.png': ImageHandler,
        '.jpg': ImageHandler,
        '.jpeg': ImageHandler,
        '.tiff': ImageHandler,
        '.tif': ImageHandler,
        '.h5': HDF5Handler,
        '.hdf5': HDF5Handler,
        '.pkl': PickleHandler,
        '.pickle': PickleHandler
    }
    
    @classmethod
    def get_handler(cls, filepath: str) -> BaseFileHandler:
        """
        Get appropriate file handler based on file extension.
        
        Args:
            filepath: Path to file
        
        Returns:
            File handler instance
        """
        ext = os.path.splitext(filepath)[1].lower()
        
        if ext not in cls._handlers:
            raise ValueError(f"Unsupported file extension: {ext}")
        
        return cls._handlers[ext]()
    
    @classmethod
    def register_handler(cls, extension: str, handler_class: type):
        """
        Register a new file handler.
        
        Args:
            extension: File extension (including dot)
            handler_class: Handler class
        """
        cls._handlers[extension] = handler_class
    
    @classmethod
    def get_supported_formats(cls) -> list:
        """Get list of supported file formats."""
        return list(cls._handlers.keys())


def save_data(filepath: str, data: np.ndarray, 
              metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Convenience function to save data using appropriate handler.
    
    Args:
        filepath: Path to save file
        data: Data to save
        metadata: Optional metadata
    """
    handler = FileHandlerFactory.get_handler(filepath)
    handler.save(filepath, data, metadata)


def load_data(filepath: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Convenience function to load data using appropriate handler.
    
    Args:
        filepath: Path to load file
    
    Returns:
        Tuple of (data, metadata)
    """
    handler = FileHandlerFactory.get_handler(filepath)
    return handler.load(filepath)
