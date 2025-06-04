#!/usr/bin/env python3
"""
Parking Lot Simulation System
Generates realistic parking lot scenarios for testing the parking detection system
"""

import cv2
import numpy as np
import random
import time
import threading
import queue
import math
import os
from typing import List, Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from enum import Enum
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CarColor(Enum):
    """Available car colors"""
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)
    BLACK = (50, 50, 50)
    WHITE = (255, 255, 255)
    SILVER = (192, 192, 192)
    YELLOW = (0, 255, 255)
    ORANGE = (0, 165, 255)

@dataclass
class ParkingSpace:
    """Represents a parking space"""
    id: int
    x: int
    y: int
    width: int
    height: int
    angle: float = 0.0
    is_occupied: bool = False
    car_color: Optional[CarColor] = None
    occupation_time: float = 0.0

class ParkingLotSimulator:
    """Main parking lot simulator class"""
    
    def __init__(self, width: int = 1200, height: int = 800):
        self.width = width
        self.height = height
        self.background = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background
        self.parking_spaces: List[ParkingSpace] = []
        self.current_frame = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Simulation parameters
        self.occupation_probability = 0.7  # Probability of a space being occupied
        self.change_probability = 0.02    # Probability of occupancy change per frame
        self.fps = 10
        
        # Initialize parking lot layout
        self._create_parking_layout()
        self._draw_base_scene()
        
    def _create_parking_layout(self):
        """Create a realistic parking lot layout"""
        space_id = 0
        
        # Create horizontal parking rows
        rows = [
            {"y": 150, "spaces": 8, "angle": 0},
            {"y": 250, "spaces": 8, "angle": 0},
            {"y": 400, "spaces": 8, "angle": 0},
            {"y": 500, "spaces": 8, "angle": 0},
        ]
        
        space_width = 100
        space_height = 80
        start_x = 100
        
        for row in rows:
            for i in range(row["spaces"]):
                x = start_x + i * (space_width + 20)
                space = ParkingSpace(
                    id=space_id,
                    x=x,
                    y=row["y"],
                    width=space_width,
                    height=space_height,
                    angle=row["angle"]
                )
                self.parking_spaces.append(space)
                space_id += 1
        
        # Create angled parking spaces
        angled_rows = [
            {"y": 650, "spaces": 6, "angle": 30},
        ]
        
        for row in angled_rows:
            for i in range(row["spaces"]):
                x = start_x + i * (space_width + 10)
                space = ParkingSpace(
                    id=space_id,
                    x=x,
                    y=row["y"],
                    width=space_width,
                    height=space_height,
                    angle=row["angle"]
                )
                self.parking_spaces.append(space)
                space_id += 1
        
        logger.info(f"Created {len(self.parking_spaces)} parking spaces")
    
    def _draw_base_scene(self):
        """Draw the base parking lot scene"""
        # Fill background with asphalt color
        self.background[:] = (70, 70, 70)
        
        # Draw parking lot lines
        line_color = (255, 255, 255)  # White lines
        line_thickness = 2
        
        # Draw horizontal divider lines
        cv2.line(self.background, (0, 300), (self.width, 300), line_color, line_thickness)
        cv2.line(self.background, (0, 550), (self.width, 550), line_color, line_thickness)
        
        # Draw vertical lane lines
        for x in range(50, self.width, 150):
            cv2.line(self.background, (x, 0), (x, self.height), line_color, 1)
        
        # Draw parking space boundaries
        for space in self.parking_spaces:
            self._draw_parking_space_lines(space)
        
        # Add some background elements
        self._add_background_elements()
    
    def _draw_parking_space_lines(self, space: ParkingSpace):
        """Draw parking space boundary lines"""
        line_color = (255, 255, 255)
        line_thickness = 2
        
        if space.angle == 0:
            # Straight parking space
            # Draw rectangle
            top_left = (space.x, space.y)
            bottom_right = (space.x + space.width, space.y + space.height)
            cv2.rectangle(self.background, top_left, bottom_right, line_color, line_thickness)
            
            # Draw space number
            text_pos = (space.x + 10, space.y + 20)
            cv2.putText(self.background, str(space.id + 1), text_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)
        else:
            # Angled parking space
            self._draw_angled_space(space, line_color, line_thickness)
    
    def _draw_angled_space(self, space: ParkingSpace, color: Tuple[int, int, int], thickness: int):
        """Draw angled parking space"""
        angle_rad = math.radians(space.angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Calculate corner points
        corners = [
            (0, 0),
            (space.width, 0),
            (space.width, space.height),
            (0, space.height)
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for x, y in corners:
            new_x = int(x * cos_angle - y * sin_angle + space.x)
            new_y = int(x * sin_angle + y * cos_angle + space.y)
            rotated_corners.append((new_x, new_y))
        
        # Draw the rotated rectangle
        points = np.array(rotated_corners, np.int32)
        cv2.polylines(self.background, [points], True, color, thickness)
        
        # Add space number
        center_x = space.x + space.width // 2
        center_y = space.y + space.height // 2
        cv2.putText(self.background, str(space.id + 1), (center_x - 10, center_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _add_background_elements(self):
        """Add realistic background elements"""
        # Add some random cracks or imperfections
        for _ in range(20):
            x1, y1 = random.randint(0, self.width), random.randint(0, self.height)
            x2, y2 = x1 + random.randint(-30, 30), y1 + random.randint(-30, 30)
            cv2.line(self.background, (x1, y1), (x2, y2), (50, 50, 50), 1)
        
        # Add some oil stains
        for _ in range(10):
            center = (random.randint(50, self.width-50), random.randint(50, self.height-50))
            radius = random.randint(10, 25)
            cv2.circle(self.background, center, radius, (30, 30, 30), -1)
    
    def _draw_car(self, image: np.ndarray, space: ParkingSpace):
        """Draw a car in the parking space"""
        if not space.is_occupied or space.car_color is None:
            return
        
        car_color = space.car_color.value
        
        # Car dimensions (slightly smaller than parking space)
        car_width = int(space.width * 0.8)
        car_height = int(space.height * 0.7)
        
        # Car position (centered in parking space)
        car_x = space.x + (space.width - car_width) // 2
        car_y = space.y + (space.height - car_height) // 2
        
        if space.angle == 0:
            # Draw straight car
            self._draw_straight_car(image, car_x, car_y, car_width, car_height, car_color)
        else:
            # Draw angled car
            self._draw_angled_car(image, space, car_width, car_height, car_color)
    
    def _draw_straight_car(self, image: np.ndarray, x: int, y: int, width: int, height: int, color: Tuple[int, int, int]):
        """Draw a straight (non-rotated) car"""
        # Main car body
        cv2.rectangle(image, (x, y), (x + width, y + height), color, -1)
        
        # Car outline
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 0), 2)
        
        # Windows (lighter color)
        window_color = tuple(min(255, c + 50) for c in color)
        window_margin = 10
        cv2.rectangle(image, (x + window_margin, y + window_margin), 
                     (x + width - window_margin, y + height - window_margin), window_color, -1)
        
        # Add some car details
        self._add_car_details(image, x, y, width, height, color)
    
    def _draw_angled_car(self, image: np.ndarray, space: ParkingSpace, car_width: int, car_height: int, color: Tuple[int, int, int]):
        """Draw an angled car"""
        angle_rad = math.radians(space.angle)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)
        
        # Calculate car center
        center_x = space.x + space.width // 2
        center_y = space.y + space.height // 2
        
        # Car corners relative to center
        half_width = car_width // 2
        half_height = car_height // 2
        corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height)
        ]
        
        # Rotate and translate corners
        rotated_corners = []
        for x, y in corners:
            new_x = int(x * cos_angle - y * sin_angle + center_x)
            new_y = int(x * sin_angle + y * cos_angle + center_y)
            rotated_corners.append((new_x, new_y))
        
        # Draw car body
        points = np.array(rotated_corners, np.int32)
        cv2.fillPoly(image, [points], color)
        cv2.polylines(image, [points], True, (0, 0, 0), 2)
    
    def _add_car_details(self, image: np.ndarray, x: int, y: int, width: int, height: int, color: Tuple[int, int, int]):
        """Add realistic car details"""
        # Headlights
        light_color = (255, 255, 200)
        light_radius = 5
        cv2.circle(image, (x + 15, y + height - 15), light_radius, light_color, -1)
        cv2.circle(image, (x + width - 15, y + height - 15), light_radius, light_color, -1)
        
        # Door lines
        door_color = tuple(max(0, c - 30) for c in color)
        cv2.line(image, (x + width//3, y), (x + width//3, y + height), door_color, 1)
        cv2.line(image, (x + 2*width//3, y), (x + 2*width//3, y + height), door_color, 1)
    
    def update_simulation(self):
        """Update the simulation state"""
        current_time = time.time()
        
        for space in self.parking_spaces:
            # Random occupancy changes
            if random.random() < self.change_probability:
                if space.is_occupied:
                    # Car might leave
                    if current_time - space.occupation_time > 5.0:  # Minimum 5 seconds
                        space.is_occupied = False
                        space.car_color = None
                        space.occupation_time = 0.0
                else:
                    # Car might arrive
                    space.is_occupied = True
                    space.car_color = random.choice(list(CarColor))
                    space.occupation_time = current_time
        
        # Initial random occupation
        if not hasattr(self, '_initialized'):
            for space in self.parking_spaces:
                if random.random() < self.occupation_probability:
                    space.is_occupied = True
                    space.car_color = random.choice(list(CarColor))
                    space.occupation_time = current_time
            self._initialized = True
    
    def generate_frame(self) -> np.ndarray:
        """Generate a single frame of the simulation"""
        # Start with base background
        frame = self.background.copy()
        
        # Update simulation state
        self.update_simulation()
        
        # Draw cars
        for space in self.parking_spaces:
            self._draw_car(frame, space)
        
        # Add some realistic noise and lighting effects
        frame = self._add_realistic_effects(frame)
        
        return frame
    
    def _add_realistic_effects(self, frame: np.ndarray) -> np.ndarray:
        """Add realistic camera effects"""
        # Add slight gaussian blur to simulate camera quality
        frame = cv2.GaussianBlur(frame, (3, 3), 0.5)
        
        # Add slight noise
        noise = np.random.normal(0, 5, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)
        
        # Adjust brightness and contrast slightly
        alpha = random.uniform(0.9, 1.1)  # Contrast
        beta = random.randint(-10, 10)    # Brightness
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        
        # Add timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, f"Simulation: {timestamp}", (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def get_parking_rois(self) -> List[List[int]]:
        """Get parking space ROIs for the detection system"""
        rois = []
        for space in self.parking_spaces:
            if space.angle == 0:
                # Straight parking space
                roi = [space.x, space.y, space.x + space.width, space.y + space.height]
            else:
                # For angled spaces, use bounding box
                angle_rad = math.radians(space.angle)
                cos_angle = math.cos(angle_rad)
                sin_angle = math.sin(angle_rad)
                
                # Calculate rotated corners
                corners = [
                    (0, 0), (space.width, 0), 
                    (space.width, space.height), (0, space.height)
                ]
                
                rotated_corners = []
                for x, y in corners:
                    new_x = x * cos_angle - y * sin_angle + space.x
                    new_y = x * sin_angle + y * cos_angle + space.y
                    rotated_corners.append((new_x, new_y))
                
                # Get bounding box
                x_coords = [p[0] for p in rotated_corners]
                y_coords = [p[1] for p in rotated_corners]
                roi = [int(min(x_coords)), int(min(y_coords)), 
                       int(max(x_coords)), int(max(y_coords))]
            
            rois.append(roi)
        
        return rois
    
    def start_simulation(self, display: bool = True):
        """Start the simulation"""
        self.is_running = True
        logger.info("Starting parking lot simulation...")
        
        if display:
            cv2.namedWindow('Parking Lot Simulation', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Parking Lot Simulation', self.width, self.height)
        
        try:
            while self.is_running:
                frame = self.generate_frame()
                self.current_frame = frame.copy()
                
                # Add frame to queue for external access
                if not self.frame_queue.full():
                    self.frame_queue.put(frame.copy())
                
                if display:
                    # Add instructions
                    info_frame = frame.copy()
                    cv2.putText(info_frame, "Press 'q' to quit, 'r' to randomize occupancy", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(info_frame, f"Occupied: {sum(1 for s in self.parking_spaces if s.is_occupied)}/{len(self.parking_spaces)}", 
                              (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Parking Lot Simulation', info_frame)
                
                key = cv2.waitKey(int(1000/self.fps)) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Randomize all parking spaces
                    current_time = time.time()
                    for space in self.parking_spaces:
                        if random.random() < 0.5:
                            space.is_occupied = True
                            space.car_color = random.choice(list(CarColor))
                            space.occupation_time = current_time
                        else:
                            space.is_occupied = False
                            space.car_color = None
                            space.occupation_time = 0.0
                    logger.info("Randomized parking occupancy")
        
        finally:
            self.is_running = False
            if display:
                cv2.destroyAllWindows()
            logger.info("Simulation stopped")
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame (for external access)"""
        if not self.frame_queue.empty():
            return self.frame_queue.get()
        return self.current_frame

class SimulatedCamera:
    """Simulated camera that provides frames from the parking lot simulation"""
    
    def __init__(self, simulator: ParkingLotSimulator):
        self.simulator = simulator
        self.is_opened = True
        
    def isOpened(self) -> bool:
        """Check if camera is opened"""
        return self.is_opened and self.simulator.is_running
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the simulation"""
        if not self.is_opened or not self.simulator.is_running:
            return False, None
        
        frame = self.simulator.get_frame()
        if frame is not None:
            return True, frame
        else:
            # Generate a new frame if none available
            frame = self.simulator.generate_frame()
            return True, frame
    
    def release(self):
        """Release the camera"""
        self.is_opened = False
        self.simulator.is_running = False

def integrate_with_detection_system():
    """Integration example with the parking detection system"""
    try:
        # Import the detection system (assuming it's in the same directory)
        from parking_detection_system import ParkingSpaceDetector, RealTimeParkingDetector
        
        # Create simulator
        simulator = ParkingLotSimulator()
        
        # Get predefined ROIs from simulator
        parking_rois = simulator.get_parking_rois()
        
        # Start simulation in background thread
        sim_thread = threading.Thread(target=simulator.start_simulation, args=(False,))
        sim_thread.daemon = True
        sim_thread.start()
        
        # Wait for simulation to start
        time.sleep(2)
        
        # Load trained model (you need to have a trained model)
        detector = ParkingSpaceDetector()
        try:
            detector.load_model()
            logger.info("Loaded trained model successfully")
        except FileNotFoundError:
            logger.error("No trained model found. Please train a model first.")
            return
        
        # Create real-time detector with predefined ROIs
        rt_detector = RealTimeParkingDetector(detector, parking_rois)
        
        # Create simulated camera
        sim_camera = SimulatedCamera(simulator)
        
        # Run detection on simulated feed
        logger.info("Starting detection on simulated camera feed...")
        cv2.namedWindow('Parking Detection - Simulation')
        
        while sim_camera.isOpened():
            ret, frame = sim_camera.read()
            if not ret:
                break
            
            try:
                # Predict parking spaces
                predictions, probabilities = detector.predict_spaces(frame, parking_rois)
                
                # Visualize results
                result_frame = detector.visualize_results(frame, parking_rois, predictions)
                
                # Add simulation info
                cv2.putText(result_frame, "Simulated Camera Feed", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(result_frame, f"Detected Empty: {predictions.count(0)}/{len(predictions)}", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Parking Detection - Simulation', result_frame)
                
            except Exception as e:
                logger.error(f"Detection error: {e}")
                cv2.imshow('Parking Detection - Simulation', frame)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                break
        
        sim_camera.release()
        simulator.is_running = False
        cv2.destroyAllWindows()
        
    except ImportError:
        logger.error("Could not import parking detection system. Make sure parking_detection_system.py is available.")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Parking Lot Simulation System')
    parser.add_argument('--mode', choices=['simulate', 'integrate'], default='simulate',
                       help='Mode: simulate (standalone) or integrate (with detection system)')
    parser.add_argument('--width', type=int, default=1200, help='Simulation width')
    parser.add_argument('--height', type=int, default=800, help='Simulation height')
    parser.add_argument('--fps', type=int, default=10, help='Simulation FPS')
    
    args = parser.parse_args()
    
    if args.mode == 'simulate':
        # Run standalone simulation
        simulator = ParkingLotSimulator(args.width, args.height)
        simulator.fps = args.fps
        
        # Print ROI information
        rois = simulator.get_parking_rois()
        logger.info(f"Generated {len(rois)} parking space ROIs")
        logger.info("ROIs can be used with the detection system:")
        for i, roi in enumerate(rois[:5]):  # Show first 5
            logger.info(f"  Space {i+1}: {roi}")
        
        simulator.start_simulation()
        
    elif args.mode == 'integrate':
        # Run integrated with detection system
        integrate_with_detection_system()

if __name__ == "__main__":
    main()