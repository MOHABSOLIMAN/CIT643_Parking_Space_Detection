# Automated Parking Space Detection System

A comprehensive computer vision system for detecting parking space occupancy using machine learning. This system leverages the PKLot dataset for training and supports real-time detection through camera feeds.

## Features

- **Dataset Support**: Compatible with PKLot dataset format
- **Machine Learning**: SVM classifier with feature engineering
- **Real-time Detection**: Live camera feed processing
- **Interactive ROI Selection**: Click-and-drag parking space selection
- **Performance Evaluation**: Comprehensive metrics and visualizations
- **Model Persistence**: Save and load trained models
- **Visualization**: Annotated output images with bounding boxes

## Requirements

- Python 3.7+
- OpenCV 4.x
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- PKLot dataset (optional, for training)

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/parking-detection-system.git
   cd parking-detection-system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker Installation

1. Build the Docker image:
   ```bash
   docker build -t parking-detector .
   ```
2. Run the container:
   ```bash
   # For training (mount your dataset)
   docker run -v /path/to/dataset:/app/dataset -v /path/to/output:/app/output parking-detector
   # For real-time detection (with camera access)
   docker run --device=/dev/video0 -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix parking-detector
   ```

## Dataset Preparation

The system uses the PKLot dataset with the following structure:
```
dataset/
└── PKLot/
    ├── cloudy/
    │   └── 2012-09-12/
    │       ├── image001.jpg
    │       ├── image001.xml
    │       └── ...
    └── ...
```
XML annotations should contain rotated rectangle coordinates for each parking space with occupancy labels.

## Usage

### Training Mode
Train a new model on the PKLot dataset:
```bash
python parking_detection_system.py --mode train --dataset /path/to/pklot/dataset --model parking_model.pkl --output output
```

### Testing Mode
Evaluate a trained model on test data:
```bash
python parking_detection_system.py --mode test --dataset /path/to/test/dataset --model parking_model.pkl --output output
```

### Real-time Detection Mode
Run real-time detection using a camera:
```bash
python parking_detection_system.py --mode realtime --model parking_model.pkl --camera 0
```

### Interactive Controls
- **Click and drag**: Select parking space regions (ROIs)
- **'r' key**: Reset all selected ROIs
- **'q' key**: Quit the application

## API Usage

### Basic Usage
```python
from parking_detection_system import ParkingSpaceDetector

# Initialize detector
detector = ParkingSpaceDetector()

# Training
image_paths, rois, labels = detector.load_pklot_dataset('/path/to/dataset')
detector.train_classifier(image_paths, rois, labels)
detector.save_model('parking_model.pkl')

# Prediction
detector.load_model('parking_model.pkl')
predictions, probabilities = detector.predict_spaces(image, parking_rois)
```

### Real-time Detection
```python
from parking_detection_system import ParkingSpaceDetector, RealTimeParkingDetector

# Load trained detector
detector = ParkingSpaceDetector()
detector.load_model('parking_model.pkl')

# Initialize real-time detector
rt_detector = RealTimeParkingDetector(detector)

# Set predefined parking regions (optional)
parking_rois = [[x1, y1, x2, y2], ...]  # List of bounding boxes
rt_detector.set_parking_regions(parking_rois)

# Start detection
rt_detector.detect_from_camera(camera_id=0)
```

## Model Architecture

### Feature Extraction
- **Statistical Features**: Mean intensity, variance, texture
- **Edge Features**: Sobel edge detection for boundary information
- **Preprocessing**: Histogram equalization, Gaussian blur

### Classification
- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Features**: 4-dimensional feature vector per parking space
- **Scaling**: StandardScaler for feature normalization
- **Class Balance**: Weighted classes to handle imbalanced data

## Output Files
- `parking_model.pkl`: Trained classifier and scaler
- `confusion_matrix.png`: Classification performance visualization
- `empty_spaces_comparison.png`: Actual vs predicted empty spaces
- `*_annotated.jpg`: Annotated images with detection results

## Performance Metrics
- Precision
- Recall
- F1-Score
- Accuracy

## Troubleshooting

### Common Issues
1. **Camera not detected**:
   ```bash
   # List available cameras
   ls /dev/video*
   # Try different camera IDs
   python parking_detection_system.py --mode realtime --camera 1
   ```
2. **XML parsing errors**:
   - Ensure XML files match the expected PKLot format
   - Check file permissions and paths
3. **Memory issues with large datasets**:
   - Process dataset in batches
   - Reduce image resolution if needed
4. **Poor detection accuracy**:
   - Collect more training data
   - Adjust feature extraction parameters
   - Try different SVM parameters

### Docker Issues
1. **Camera access in Docker**:
   ```bash
   docker run --device=/dev/video0 ...
   ```
2. **Display issues**:
   ```bash
   xhost +local:docker
   docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix ...
   ```

## Performance Optimization
### Speed Improvements
- Use smaller ROI regions
- Reduce image resolution
- Implement multi-threading for batch processing
- Consider using faster classifiers (e.g., Random Forest)

### Accuracy Improvements
- Collect more diverse training data
- Implement data augmentation
- Use deep learning models (CNN)
- Add temporal consistency for video streams

## Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you use this system in your research, please cite:
```bibtex
@software{parking_detection_system,
  title={Automated Parking Space Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/parking-detection-system}
}
```

## Acknowledgments
- PKLot dataset creators for providing the benchmark dataset
- OpenCV community for computer vision tools
- Scikit-learn developers for machine learning algorithms

## Contact
For questions and support, please open an issue on GitHub or contact [your-email@example.com].