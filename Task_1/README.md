# Detection Summary Engine - Task 1

## Overview
This project implements a **Detection Summary Engine** using YOLOv8x for object detection in video streams. The system processes every 5th frame of a 15-20 second video, generates comprehensive detection statistics, and provides visualizations for analysis.

## Features
- **YOLOv8x Integration**: Uses state-of-the-art YOLOv8 extra-large model for high accuracy
- **Frame-by-Frame Analysis**: Processes every 5th frame for optimal performance
- **JSON Output**: Structured detection data with bounding boxes and confidence scores
- **Statistical Analysis**: Object frequency counting and class diversity metrics
- **Visualization**: Automated bar chart generation for detection frequencies
- **Annotated Frames**: Optional saving of frames with bounding box overlays

## Technical Architecture

### 1. Video Processing Pipeline
```python
# Frame selection logic
if frame_count % 5 == 0:
    detections = self._detect_objects(frame, frame_count, timestamp)
```
- **Input Validation**: Checks video duration (15-20 seconds) and file integrity
- **Performance Optimization**: Processes every 5th frame to balance accuracy and speed
- **Memory Management**: Sequential frame processing to avoid memory overflow

### 2. Object Detection System
```python
results = self.model.predict(rgb_frame, conf=self.confidence_threshold, verbose=False)
```
- **Model Configuration**: YOLOv8x with configurable confidence threshold (default: 0.5)
- **Color Space Conversion**: BGR to RGB for optimal model performance
- **Tensor Processing**: Efficient extraction of coordinates and class predictions

### 3. Data Structure Design
```json
{
  "frame_number": 0,
  "detections": [
    {
      "class_name": "person",
      "bbox": {
        "x1": 229.14125061035156,
        "y1": 74.84062957763672,
        "x2": 293.11419677734375,
        "y2": 249.18994140625
      },
      "confidence": 0.8407524228096008
    }
  ]
}
```

## Installation & Setup

### Prerequisites
```bash
pip install ultralytics opencv-python matplotlib numpy pathlib
```

### Quick Start
1. **Clone or download** the detection engine script
2. **Prepare your video**: Ensure it's 15-20 seconds duration in .mp4 format
3. **Configure parameters** in the `main()` function:
   ```python
   video_path = "/path/to/your/video.mp4"
   model_name = "yolov8x.pt"
   confidence_threshold = 0.5
   save_annotated = True
   ```
4. **Run the engine**:
   ```bash
   python detection_engine.py
   ```

## Configuration Options

### Model Selection
- `yolov8n.pt` - Nano (fastest, lower accuracy)
- `yolov8s.pt` - Small (balanced)
- `yolov8m.pt` - Medium (good accuracy)
- `yolov8l.pt` - Large (high accuracy)
- `yolov8x.pt` - Extra Large (highest accuracy, slower)

### Detection Parameters
- **confidence_threshold**: Filter detections by confidence score (0.0-1.0)
- **save_annotated_frames**: Enable/disable annotated frame saving
- **output_dir**: Customize output directory name

## Output Structure
```
task1_output/
├── frame_detections.json        # Complete detection data
├── object_frequency_chart.png   # Statistical visualization
└── annotated_frames/            # Optional annotated images
    ├── frame_0000.jpg
    ├── frame_0005.jpg
    └── ...
```

## Implementation Details

### Frame Processing Logic
```python
def _detect_objects(self, frame, frame_number, timestamp):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.model.predict(rgb_frame, conf=self.confidence_threshold, verbose=False)
    # Extract detections with coordinate conversion
```

### Coordinate System
- **Format**: Absolute pixel coordinates (x1, y1, x2, y2)
- **Origin**: Top-left corner (0, 0)
- **Precision**: Full float precision maintained from model output

### Visualization Features
- **Automatic Sorting**: Classes sorted by detection frequency
- **Total Count Display**: Shows aggregate object count
- **Color Coding**: Consistent color scheme for readability
- **Export Options**: High-resolution PNG format

## Performance Characteristics

### Processing Speed
- **Frame Rate**: Processes ~20% of total frames (every 5th frame)
- **Model Speed**: YOLOv8x ~50-100ms per frame (GPU dependent)
- **Memory Usage**: Minimal due to sequential processing

### Accuracy Metrics
- **mAP@0.5**: 53.9% (YOLOv8x on COCO dataset)
- **Classes Supported**: 80 COCO classes (persons, vehicles, animals, objects)
- **Confidence Range**: 0.0-1.0 with configurable threshold

## Error Handling
- **Video Duration Validation**: Ensures 15-20 second constraint
- **File Integrity Checks**: Validates video file accessibility
- **Empty Detection Handling**: Graceful handling of frames with no detections
- **Output Directory Management**: Automatic directory creation

## Troubleshooting

### Common Issues
1. **CUDA Out of Memory**: Reduce batch size or use smaller model (yolov8s.pt)
2. **Video Duration Error**: Verify video length with media player
3. **Import Errors**: Ensure all dependencies installed via pip
4. **No Detections**: Lower confidence threshold or check video content

### Debug Mode
Enable verbose output by modifying:
```python
results = self.model.predict(rgb_frame, conf=self.confidence_threshold, verbose=True)
```

## Use Cases
- **Traffic Analysis**: Vehicle and pedestrian counting
- **Security Monitoring**: Person and object detection
- **Sports Analytics**: Player and equipment tracking
- **Wildlife Research**: Animal behavior analysis

## License & Dependencies
- **Ultralytics YOLOv8**: AGPL-3.0 License
- **OpenCV**: Apache 2.0 License
- **Matplotlib**: PSF License
- **NumPy**: BSD License

## Technical Specifications
- **Input Format**: MP4, AVI, MOV video files
- **Output Format**: JSON, PNG, JPG
- **Model Requirements**: ~165MB storage for YOLOv8x
- **Python Version**: 3.8+ recommended

---

**Author**: Computer Vision Engineer Assignment  
**Date**: June 2025  
**Version**: 1.0