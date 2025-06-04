#!/usr/bin/env python3
"""
Automated Parking Space Detection System
Uses PKLot dataset for training and real-time detection
"""

import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import pickle
import argparse
import time
from typing import List, Tuple, Dict, Optional
import threading
import queue

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParkingSpaceDetector:
    """Main class for parking space detection system"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.classifier = None
        self.scaler = None
        self.is_trained = False
        self.model_path = model_path or "parking_model.pkl"
        
    def load_pklot_dataset(self, dataset_path: str) -> Tuple[List[str], List[List], List[List]]:
        """Load PKLot dataset with images and XML annotations"""
        image_paths = []
        all_rois = []
        all_labels = []
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = os.path.join(root, file)
                    xml_path = img_path.replace('.jpg', '.xml')
                    
                    if os.path.exists(xml_path):
                        try:
                            rois, labels = self._parse_xml_annotations(xml_path)
                            if rois and labels:
                                image_paths.append(img_path)
                                all_rois.append(rois)
                                all_labels.append(labels)
                        except Exception as e:
                            logger.warning(f"Error parsing {xml_path}: {e}")
                            continue
        
        logger.info(f"Loaded {len(image_paths)} images with annotations")
        return image_paths, all_rois, all_labels
    
    def _parse_xml_annotations(self, xml_path: str) -> Tuple[List[List], List[int]]:
        """Parse XML annotations to extract parking space coordinates and labels"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            rois = []
            labels = []
            
            for space in root.findall('.//space'):
                # Get occupancy status
                occupied = space.get('occupied', '0') == '1'
                
                # Get rotated rectangle coordinates
                rotated_rect = space.find('rotatedRect')
                if rotated_rect is not None:
                    # Extract center, size, and angle
                    center_x = float(rotated_rect.find('center').get('x', 0))
                    center_y = float(rotated_rect.find('center').get('y', 0))
                    width = float(rotated_rect.find('size').get('w', 0))
                    height = float(rotated_rect.find('size').get('h', 0))
                    angle = float(rotated_rect.find('angle').get('d', 0))
                    
                    # Convert rotated rectangle to bounding box
                    bbox = self._convert_rotated_rect_to_bbox(
                        center_x, center_y, width, height, angle
                    )
                    
                    if self._validate_bbox(bbox):
                        rois.append(bbox)
                        labels.append(1 if occupied else 0)
            
            return rois, labels
            
        except Exception as e:
            logger.error(f"Error parsing XML {xml_path}: {e}")
            return [], []
    
    def _convert_rotated_rect_to_bbox(self, cx: float, cy: float, w: float, h: float, angle: float) -> List[int]:
        """Convert rotated rectangle to axis-aligned bounding box"""
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Calculate the four corners of the rotated rectangle
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Half dimensions
        hw, hh = w/2, h/2
        
        # Four corners relative to center
        corners = np.array([
            [-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]
        ])
        
        # Rotate corners
        rotated_corners = np.array([
            [cos_a * x - sin_a * y + cx, sin_a * x + cos_a * y + cy]
            for x, y in corners
        ])
        
        # Get bounding box
        x_min = int(np.min(rotated_corners[:, 0]))
        y_min = int(np.min(rotated_corners[:, 1]))
        x_max = int(np.max(rotated_corners[:, 0]))
        y_max = int(np.max(rotated_corners[:, 1]))
        
        return [x_min, y_min, x_max, y_max]
    
    def _validate_bbox(self, bbox: List[int], min_size: int = 10) -> bool:
        """Validate bounding box coordinates"""
        x1, y1, x2, y2 = bbox
        return (x2 > x1 + min_size and y2 > y1 + min_size and 
                x1 >= 0 and y1 >= 0)
    
    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess image for feature extraction"""
        original = image.copy()
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply histogram equalization
        equalized = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
        
        return original, blurred
    
    def extract_features(self, image: np.ndarray, rois: List[List]) -> np.ndarray:
        """Extract features from parking space regions"""
        features = []
        
        for roi in rois:
            x1, y1, x2, y2 = roi
            
            # Extract region of interest
            region = image[y1:y2, x1:x2]
            
            if region.size == 0:
                features.append([0, 0, 0, 0])
                continue
            
            # Statistical features
            mean_intensity = np.mean(region)
            variance = np.var(region)
            texture = np.std(region)
            
            # Edge-based features
            sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobelx**2 + sobely**2)
            edge_density = np.mean(edge_magnitude)
            
            features.append([mean_intensity, variance, edge_density, texture])
        
        return np.array(features)
    
    def train_classifier(self, image_paths: List[str], all_rois: List[List], all_labels: List[List]):
        """Train the SVM classifier"""
        logger.info("Training classifier...")
        
        all_features = []
        flat_labels = []
        
        # Extract features from all images
        for img_path, rois, labels in tqdm(zip(image_paths, all_rois, all_labels), 
                                         desc="Extracting features", total=len(image_paths)):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                    
                _, preprocessed = self.preprocess_image(image)
                features = self.extract_features(preprocessed, rois)
                
                all_features.extend(features)
                flat_labels.extend(labels)
                
            except Exception as e:
                logger.warning(f"Error processing {img_path}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No features extracted for training")
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(flat_labels)
        
        logger.info(f"Training data: {len(X)} samples, {np.sum(y==0)} empty, {np.sum(y==1)} occupied")
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train SVM classifier
        self.classifier = SVC(
            kernel='rbf',
            C=1.0,
            class_weight='balanced',
            probability=True,
            random_state=42
        )
        
        self.classifier.fit(X_scaled, y)
        self.is_trained = True
        
        logger.info("Training completed successfully")
    
    def predict_spaces(self, image: np.ndarray, rois: List[List]) -> Tuple[List[int], np.ndarray]:
        """Predict occupancy status of parking spaces"""
        if not self.is_trained:
            raise ValueError("Classifier not trained. Call train_classifier first.")
        
        _, preprocessed = self.preprocess_image(image)
        features = self.extract_features(preprocessed, rois)
        
        if len(features) == 0:
            return [], np.array([])
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.classifier.predict(features_scaled)
        probabilities = self.classifier.predict_proba(features_scaled)
        
        return predictions.tolist(), probabilities
    
    def visualize_results(self, image: np.ndarray, rois: List[List], predictions: List[int], 
                         output_path: str = None) -> np.ndarray:
        """Visualize detection results with bounding boxes"""
        result_image = image.copy()
        
        for i, (roi, pred) in enumerate(zip(rois, predictions)):
            x1, y1, x2, y2 = roi
            
            # Choose color based on prediction
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)  # Green for empty, Red for occupied
            label = "Empty" if pred == 0 else "Occupied"
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Add text label
            label_text = f"Space {i+1}: {label}"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(result_image, (x1, y1 - text_size[1] - 10), 
                         (x1 + text_size[0], y1), color, -1)
            cv2.putText(result_image, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        return result_image
    
    def save_model(self, path: str = None):
        """Save trained model and scaler"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        save_path = path or self.model_path
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, path: str = None):
        """Load trained model and scaler"""
        load_path = path or self.model_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {load_path}")
    
    def evaluate_performance(self, image_paths: List[str], all_rois: List[List], 
                           all_labels: List[List], output_dir: str = "output"):
        """Evaluate model performance and generate visualizations"""
        if not self.is_trained:
            raise ValueError("Classifier not trained")
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_true_labels = []
        all_pred_labels = []
        
        logger.info("Evaluating performance...")
        
        for img_path, rois, true_labels in tqdm(zip(image_paths, all_rois, all_labels),
                                              desc="Processing images", total=len(image_paths)):
            try:
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                predictions, _ = self.predict_spaces(image, rois)
                
                # Visualize results
                filename = os.path.splitext(os.path.basename(img_path))[0]
                output_path = os.path.join(output_dir, f"{filename}_annotated.jpg")
                self.visualize_results(image, rois, predictions, output_path)
                
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(predictions)
                
            except Exception as e:
                logger.warning(f"Error evaluating {img_path}: {e}")
                continue
        
        # Calculate metrics
        if all_true_labels and all_pred_labels:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_true_labels, all_pred_labels, average='weighted', zero_division=0
            )
            accuracy = accuracy_score(all_true_labels, all_pred_labels)
            
            logger.info(f"Overall Metrics: Precision={precision:.2f}, Recall={recall:.2f}, "
                       f"F1={f1:.2f}, Accuracy={accuracy:.4f}")
            
            # Generate visualizations
            self._generate_evaluation_plots(all_true_labels, all_pred_labels, output_dir)
            
            return {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            }
        
        return None
    
    def _generate_evaluation_plots(self, true_labels: List[int], pred_labels: List[int], output_dir: str):
        """Generate confusion matrix and comparison plots"""
        # Confusion Matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Empty', 'Occupied'], 
                   yticklabels=['Empty', 'Occupied'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Empty spaces comparison
        true_empty = np.sum(np.array(true_labels) == 0)
        pred_empty = np.sum(np.array(pred_labels) == 0)
        
        plt.figure(figsize=(8, 6))
        categories = ['Actual Empty', 'Predicted Empty']
        values = [true_empty, pred_empty]
        bars = plt.bar(categories, values, color=['skyblue', 'lightcoral'])
        plt.title('Empty Spaces Comparison')
        plt.ylabel('Number of Spaces')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'empty_spaces_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

class RealTimeParkingDetector:
    """Real-time parking detection using camera feed"""
    
    def __init__(self, detector: ParkingSpaceDetector, parking_rois: List[List] = None):
        self.detector = detector
        self.parking_rois = parking_rois or []
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        
    def set_parking_regions(self, rois: List[List]):
        """Set predefined parking space regions"""
        self.parking_rois = rois
        
    def add_parking_region(self, roi: List[int]):
        """Add a single parking region"""
        self.parking_rois.append(roi)
        
    def detect_from_camera(self, camera_id: int = 0, display: bool = True):
        """Real-time detection from camera feed"""
        if not self.detector.is_trained:
            raise ValueError("Detector not trained. Load or train a model first.")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Cannot open camera {camera_id}")
        
        logger.info("Starting real-time detection. Press 'q' to quit, 'r' to reset ROIs")
        self.is_running = True
        
        # Mouse callback for ROI selection
        roi_selector = ROISelector()
        if display:
            cv2.namedWindow('Parking Detection')
            cv2.setMouseCallback('Parking Detection', roi_selector.mouse_callback)
        
        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Use predefined ROIs or selected ROIs
                current_rois = self.parking_rois if self.parking_rois else roi_selector.rois
                
                if current_rois:
                    try:
                        predictions, probabilities = self.detector.predict_spaces(frame, current_rois)
                        result_frame = self.detector.visualize_results(frame, current_rois, predictions)
                        
                        # Add instructions
                        cv2.putText(result_frame, "Press 'q' to quit, 'r' to reset ROIs", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(result_frame, f"Spaces: {len(current_rois)}, Empty: {predictions.count(0)}", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                    except Exception as e:
                        logger.error(f"Detection error: {e}")
                        result_frame = frame.copy()
                else:
                    result_frame = frame.copy()
                    cv2.putText(result_frame, "Click and drag to select parking spaces", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw current selection
                roi_selector.draw_current_selection(result_frame)
                roi_selector.draw_existing_rois(result_frame)
                
                if display:
                    cv2.imshow('Parking Detection', result_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    roi_selector.reset_rois()
                    logger.info("ROIs reset")
                
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
            self.is_running = False

class ROISelector:
    """Helper class for selecting parking space regions interactively"""
    
    def __init__(self):
        self.rois = []
        self.drawing = False
        self.start_point = None
        self.current_rect = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.current_rect = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.current_rect = (self.start_point[0], self.start_point[1], x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.start_point:
                x1, y1 = self.start_point
                x2, y2 = x, y
                
                # Ensure proper rectangle coordinates
                roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
                
                # Add ROI if it's large enough
                if abs(x2 - x1) > 20 and abs(y2 - y1) > 20:
                    self.rois.append(roi)
                    logger.info(f"Added ROI: {roi}")
                
                self.drawing = False
                self.current_rect = None
    
    def draw_current_selection(self, image):
        """Draw the currently being selected rectangle"""
        if self.current_rect:
            x1, y1, x2, y2 = self.current_rect
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 0), 2)
    
    def draw_existing_rois(self, image):
        """Draw existing ROIs"""
        for i, roi in enumerate(self.rois):
            x1, y1, x2, y2 = roi
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(image, f'ROI {i+1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    def reset_rois(self):
        """Reset all ROIs"""
        self.rois = []

def main():
    """Main function to run the parking detection system"""
    parser = argparse.ArgumentParser(description='Parking Space Detection System')
    parser.add_argument('--mode', choices=['train', 'test', 'realtime'], default='train',
                       help='Mode of operation')
    parser.add_argument('--dataset', type=str, help='Path to PKLot dataset')
    parser.add_argument('--model', type=str, default='parking_model.pkl',
                       help='Path to model file')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID for real-time detection')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ParkingSpaceDetector(args.model)
    
    if args.mode == 'train':
        if not args.dataset:
            raise ValueError("Dataset path required for training")
        
        # Load dataset and train
        image_paths, all_rois, all_labels = detector.load_pklot_dataset(args.dataset)
        detector.train_classifier(image_paths, all_rois, all_labels)
        detector.save_model()
        
        # Evaluate performance
        metrics = detector.evaluate_performance(image_paths, all_rois, all_labels, args.output)
        if metrics:
            print(f"Training completed. Metrics: {metrics}")
    
    elif args.mode == 'test':
        if not args.dataset:
            raise ValueError("Dataset path required for testing")
        
        # Load model and test
        detector.load_model()
        image_paths, all_rois, all_labels = detector.load_pklot_dataset(args.dataset)
        metrics = detector.evaluate_performance(image_paths, all_rois, all_labels, args.output)
        if metrics:
            print(f"Testing completed. Metrics: {metrics}")
    
    elif args.mode == 'realtime':
        # Load model and start real-time detection
        detector.load_model()
        rt_detector = RealTimeParkingDetector(detector)
        rt_detector.detect_from_camera(args.camera)

if __name__ == "__main__":
    main()
