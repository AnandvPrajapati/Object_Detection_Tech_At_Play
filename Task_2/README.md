# Real-Time Stream Simulation & Event Trigger - Task 2

## Overview
This implementation simulates real-time video stream processing for crowd detection using YOLOv8x model. The system processes every 3rd frame, monitors for crowd events (3+ people), and triggers alerts when crowds persist across 5 consecutive frames.

## Technical Architecture

### 1. Stream Simulation Pipeline
```python
# Core processing loop - every 3rd frame
if frame_idx % 3 == 0:
    timestamp = frame_idx / fps
    people_count, detections = self._detect_people(frame)
    alert_triggered = self._check_crowd_alert(people_count, timestamp, frame_idx)
```

**Key Design Decisions:**
- **Frame Sampling**: Processes every 3rd frame to simulate real-time constraints
- **Temporal Analysis**: Uses sliding window approach with `deque(maxlen=5)`
- **Alert Hysteresis**: Prevents duplicate alerts with 15-frame cooldown period

### 2. YOLOv8 Integration Strategy
```python
self.model = YOLO(model_path)
self.model.conf = confidence_threshold
self.person_class_id = 0  # COCO dataset person class
```

**Detection Implementation:**
- **Model Configuration**: Uses YOLOv8x for maximum accuracy
- **Class Filtering**: Specifically targets person class (ID: 0) from COCO dataset
- **Confidence Handling**: Configurable threshold for detection sensitivity
- **Tensor Processing**: Converts PyTorch tensors to NumPy for coordinate extraction

### 3. Crowd Alert Logic
```python
self.consecutive_crowd_frames.append(count >= 3)
if len(self.consecutive_crowd_frames) == 5 and all(self.consecutive_crowd_frames):
    # Trigger alert with cooldown mechanism
```

**Alert Trigger Conditions:**
1. **Crowd Threshold**: 3 or more people detected
2. **Persistence Check**: Condition must hold for 5 consecutive processed frames
3. **Cooldown Period**: 15-frame gap prevents duplicate alerts
4. **Temporal Tracking**: Maintains frame-by-frame crowd state history

### 4. Real-Time Overlay System
```python
def _create_overlay(self, frame, detections, count, alert):
    # Dynamic color coding: red for alerts, green for normal
    color = (0, 0, 255) if alert else (0, 255, 0)
    # Real-time status indicators
    cv2.putText(overlay, f"People Count: {count}", (20, 40), ...)
```

**Visual Feedback Features:**
- **Dynamic Bounding Boxes**: Color-coded based on alert state
- **Live Counters**: Real-time people count display
- **Alert Notifications**: Visual alert overlays during crowd events
- **Consecutive Frame Tracking**: Progress indicator for alert conditions

## Implementation Details

### Performance Optimization
- **Memory Management**: Uses `deque` with fixed size for efficient sliding window
- **Processing Efficiency**: 3-frame interval balances accuracy vs. speed
- **GPU Acceleration**: Automatic CUDA utilization when available
- **Selective Video Writing**: Optional overlay saving reduces I/O overhead

### Data Structures
```python
alert = {
    'alert_id': self.alert_count,
    'timestamp': timestamp,
    'frame_number': frame_number,
    'people_count': count,
    'message': f"Crowd Detected: {count} people",
    'datetime': datetime.now().isoformat()
}
```

**Alert Metadata:**
- **Unique Identification**: Sequential alert IDs
- **Temporal Information**: Frame numbers and timestamps
- **Detection Details**: People count and confidence scores
- **Human-Readable Messages**: Formatted alert descriptions

### Error Handling & Validation
- **Video File Validation**: Checks successful video opening
- **Frame Processing**: Handles end-of-stream conditions
- **Resource Management**: Proper cleanup of video readers/writers
- **Empty Detection Handling**: Graceful handling of no-detection scenarios

## Configuration Options

### Model Parameters
```python
simulator = RealTimeStreamSimulatorYOLOv8(
    model_path='yolov8x.pt',           # Model variant selection
    confidence_threshold=0.5            # Detection sensitivity
)
```

**Available Models:**
- `yolov8n.pt` - Fastest inference (~2-3ms)
- `yolov8s.pt` - Balanced performance (~5-7ms)
- `yolov8m.pt` - Higher accuracy (~10-15ms)
- `yolov8l.pt` - Premium accuracy (~15-20ms)
- `yolov8x.pt` - Maximum accuracy (~20-25ms)

### Processing Parameters
```python
simulator.simulate_stream(
    video_path="input_video.mp4",       # Input video file
    output_dir="task2_outputs",         # Output directory
    save_overlay=True                   # Enable annotated video
)
```

## Installation & Dependencies

### Required Packages
```bash
pip install ultralytics opencv-python matplotlib numpy
```

### GPU Support (Optional)
```bash
# For CUDA acceleration
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage Instructions

### Basic Usage
```python
# Initialize simulator
simulator = RealTimeStreamSimulatorYOLOv8(
    model_path='yolov8x.pt',
    confidence_threshold=0.5
)

# Process video stream
alerts = simulator.simulate_stream(
    video_path="your_video.mp4",
    output_dir="outputs",
    save_overlay=True
)
```

### Advanced Configuration
```python
# Custom confidence threshold for sensitive environments
simulator = RealTimeStreamSimulatorYOLOv8(
    model_path='yolov8s.pt',      # Faster model for real-time
    confidence_threshold=0.3       # Lower threshold for better recall
)

# Process with custom output settings
simulator.simulate_stream(
    video_path="security_feed.mp4",
    output_dir="security_alerts",
    save_overlay=False            # Skip video overlay for performance
)
```

## Output Structure

```
task2_outputs/
├── alert_logs.json              # Structured alert data
├── alert_logs.txt               # Human-readable alert log
├── alert_timeline.png           # Temporal visualization
└── stream_with_alerts.mp4       # Annotated video (optional)
```

### JSON Output Format
```json
{
  "total_alerts": 3,
  "alerts": [
    {
      "alert_id": 1,
      "timestamp": 15.67,
      "frame_number": 470,
      "people_count": 4,
      "message": "Crowd Detected: 4 people",
      "datetime": "2025-06-08T13:45:23.123456"
    }
  ]
}
```

## Performance Characteristics

### Processing Speed
- **YOLOv8n**: ~100-150 FPS (real-time capable)
- **YOLOv8s**: ~50-80 FPS (near real-time)
- **YOLOv8x**: ~20-30 FPS (high-accuracy batch processing)

### Memory Usage
- **Base Model**: 50-200MB depending on variant
- **Frame Buffer**: Minimal due to streaming approach
- **Alert Storage**: Scales linearly with alert frequency

### Accuracy Metrics
- **Person Detection mAP**: 67.2% (YOLOv8x on COCO)
- **False Positive Rate**: Reduced through consecutive frame filtering
- **Alert Precision**: Enhanced by cooldown mechanism

## Troubleshooting

### Common Issues
1. **Model Download**: First run downloads model automatically
2. **CUDA Errors**: Falls back to CPU if GPU unavailable
3. **Video Codec**: Use MP4 format for best compatibility
4. **Memory Issues**: Reduce model size or increase system RAM

### Performance Tuning
- **Real-time Processing**: Use YOLOv8n with confidence 0.3-0.4
- **Accuracy Priority**: Use YOLOv8x with confidence 0.5-0.7
- **Balanced Mode**: Use YOLOv8s with confidence 0.4-0.6

## Technical Notes

### Frame Rate Considerations
The system processes every 3rd frame, effectively operating at 1/3 of the original video frame rate. For a 30 FPS video, the system analyzes ~10 FPS, which provides sufficient temporal resolution for crowd detection while maintaining computational efficiency.

### Alert Sensitivity Tuning
The 5-consecutive-frame requirement corresponds to approximately 1.5 seconds of real-time processing (5 frames × 3-frame interval ÷ 30 FPS). This duration balances between rapid response and false positive reduction.

### Scalability Considerations
The current implementation is designed for single-stream processing. For multi-stream scenarios, consider:
- Separate model instances per stream
- Shared model with thread-safe inference
- Distributed processing across multiple devices