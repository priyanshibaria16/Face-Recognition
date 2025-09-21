# Simple Face Recognition Attendance

A minimal, accurate face recognition attendance system in a single file.

## Features
- Improved face recognition accuracy using multiple feature extraction methods
- Real-time attendance tracking
- Simple GUI interface
- Import/Export functionality
- SQLite database storage

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python face_recognition_app.py
   ```

## Usage

1. **Add Person**: Click "Add Person", enter name, capture face images with SPACE key
2. **Import Images**: Click "Import Images" to add faces from files
3. **Start Recognition**: Click "Start Recognition" to begin attendance tracking
4. **View Records**: Click "View Records" to see attendance history
5. **Export**: Click "Export Attendance" to save records as CSV

## Project Structure

```
simple-face-recognition/
├── face_recognition_app.py    # Main application (single file)
├── requirements.txt           # 4 essential dependencies
├── README.md                 # This file
└── data/                     # Auto-created data storage
    ├── faces/                # Face images
    ├── face_data.json        # Face encodings
    └── attendance.db         # SQLite database
```

## Improvements Made

- **Better Accuracy**: Uses LBP (Local Binary Pattern) + Histogram + Gradient features
- **Optimized Detection**: Improved face detection parameters
- **Single File**: All functionality in one file for maximum simplicity
- **Minimal Dependencies**: Only 4 essential packages
- **Better Recognition**: Higher threshold and multiple feature comparison

## Dependencies

- `opencv-python` - Computer vision
- `numpy` - Numerical computations  
- `pandas` - Data handling
- `Pillow` - Image processing

Total project: **1 main file + 4 dependencies**