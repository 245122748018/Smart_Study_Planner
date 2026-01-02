"""
Focus Detection Core Module

This module contains the core focus detection logic using MediaPipe Face Mesh
for real-time student attention tracking.
"""

import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional
import time


@dataclass
class FocusMetrics:
    """Data class to store all focus-related metrics."""
    
    # Eye metrics
    left_ear: float = 0.0  # Left Eye Aspect Ratio
    right_ear: float = 0.0  # Right Eye Aspect Ratio
    avg_ear: float = 0.0  # Average Eye Aspect Ratio
    
    # Mouth metrics
    mar: float = 0.0  # Mouth Aspect Ratio
    
    # Gaze metrics
    gaze_horizontal: float = 0.0  # -1 (left) to 1 (right)
    gaze_vertical: float = 0.0  # -1 (down) to 1 (up)
    
    # Head pose metrics
    head_horizontal: float = 0.0  # -1 (left) to 1 (right)
    head_vertical: float = 0.0  # -1 (down) to 1 (up)
    
    # Detection flags
    is_blinking: bool = False
    is_yawning: bool = False
    looking_at_screen: bool = True
    head_centered: bool = True
    
    # Focus score
    focus_score: int = 100
    
    # Session statistics
    total_blinks: int = 0
    blink_rate: float = 0.0  # blinks per minute


class FocusDetector:
    """
    Focus detection system using MediaPipe Face Mesh for real-time monitoring.
    
    Features:
    - Face detection and landmark tracking
    - Eye Aspect Ratio (EAR) for blink detection
    - Mouth Aspect Ratio (MAR) for yawn detection
    - Gaze tracking using iris position
    - Head pose estimation
    - Focus score calculation (0-100)
    """
    
    # MediaPipe Face Mesh landmark indices
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    LEFT_IRIS_INDICES = [468, 469, 470, 471, 472]
    RIGHT_IRIS_INDICES = [473, 474, 475, 476, 477]
    MOUTH_INDICES = [61, 291, 0, 17, 269, 405]
    NOSE_TIP_INDEX = 1
    FACE_OVAL_INDICES = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
                          397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                          172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    # Detection thresholds
    EAR_BLINK_THRESHOLD = 0.21
    MAR_YAWN_THRESHOLD = 0.6
    GAZE_THRESHOLD = 0.3  # Threshold for looking away
    HEAD_POSE_THRESHOLD = 0.25  # Threshold for head not centered
    
    def __init__(self):
        """Initialize the FocusDetector with MediaPipe Face Mesh."""
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Metrics tracking
        self.metrics = FocusMetrics()
        self.blink_counter = 0
        self.session_start_time = time.time()
        self.last_blink_time = time.time()
        
    def calculate_ear(self, eye_landmarks: list) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection.
        
        Args:
            eye_landmarks: List of 6 eye landmark points
            
        Returns:
            Eye Aspect Ratio value
        """
        # Vertical eye distances
        vertical_1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        vertical_2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal eye distance
        horizontal = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # EAR formula
        ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return ear
    
    def calculate_mar(self, mouth_landmarks: list) -> float:
        """
        Calculate Mouth Aspect Ratio (MAR) for yawn detection.
        
        Args:
            mouth_landmarks: List of 6 mouth landmark points
            
        Returns:
            Mouth Aspect Ratio value
        """
        # Vertical mouth distances
        vertical_1 = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[5])
        vertical_2 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[4])
        
        # Horizontal mouth distance
        horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[3])
        
        # MAR formula
        mar = (vertical_1 + vertical_2) / (2.0 * horizontal)
        return mar
    
    def calculate_gaze(self, iris_center: np.ndarray, eye_corners: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
        """
        Calculate gaze direction based on iris position.
        
        Args:
            iris_center: Center point of iris
            eye_corners: Tuple of (left_corner, right_corner) of eye
            
        Returns:
            Tuple of (horizontal_gaze, vertical_gaze) normalized to [-1, 1]
        """
        left_corner, right_corner = eye_corners
        
        # Calculate horizontal position relative to eye width
        eye_width = np.linalg.norm(right_corner - left_corner)
        horizontal_offset = iris_center[0] - left_corner[0]
        horizontal_ratio = (horizontal_offset / eye_width) * 2 - 1  # Normalize to [-1, 1]
        
        # Calculate vertical position (simplified)
        vertical_ratio = 0.0  # Placeholder for vertical gaze
        
        return horizontal_ratio, vertical_ratio
    
    def calculate_head_pose(self, landmarks: list, image_shape: Tuple[int, int]) -> Tuple[float, float]:
        """
        Calculate head pose based on face landmarks.
        
        Args:
            landmarks: All face landmarks
            image_shape: Shape of the image (height, width)
            
        Returns:
            Tuple of (horizontal_pose, vertical_pose) normalized to [-1, 1]
        """
        h, w = image_shape[:2]
        
        # Get nose tip position
        nose_tip = landmarks[self.NOSE_TIP_INDEX]
        
        # Calculate face bounding box center
        face_points = [landmarks[i] for i in self.FACE_OVAL_INDICES]
        face_x = np.mean([p[0] for p in face_points])
        face_y = np.mean([p[1] for p in face_points])
        
        # Calculate relative position
        horizontal_pose = (nose_tip[0] - face_x) / (w * 0.1)  # Normalize
        vertical_pose = (nose_tip[1] - face_y) / (h * 0.1)  # Normalize
        
        # Clamp to [-1, 1]
        horizontal_pose = max(-1, min(1, horizontal_pose))
        vertical_pose = max(-1, min(1, vertical_pose))
        
        return horizontal_pose, vertical_pose
    
    def calculate_focus_score(self) -> int:
        """
        Calculate focus score based on current metrics.
        
        Returns:
            Focus score from 0-100
        """
        score = 100
        
        # Penalty for not looking at screen
        if not self.metrics.looking_at_screen:
            score -= 30
        
        # Penalty for head not centered
        if not self.metrics.head_centered:
            score -= 20
        
        # Penalty for yawning
        if self.metrics.is_yawning:
            score -= 15
        
        # Penalty for high blink rate (fatigue)
        if self.metrics.blink_rate > 30:  # More than 30 blinks per minute
            score -= 10
        
        # Penalty for very low blink rate (zoning out)
        if self.metrics.blink_rate < 5 and time.time() - self.session_start_time > 60:
            score -= 5
        
        return max(0, score)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, FocusMetrics]:
        """
        Process a video frame and detect focus metrics.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, metrics)
        """
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w, _ = frame.shape
            
            # Extract landmark coordinates
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append(np.array([lm.x * w, lm.y * h, lm.z]))
            
            # Calculate Eye Aspect Ratio
            left_eye = np.array([landmarks[i][:2] for i in self.LEFT_EYE_INDICES])
            right_eye = np.array([landmarks[i][:2] for i in self.RIGHT_EYE_INDICES])
            
            self.metrics.left_ear = self.calculate_ear(left_eye)
            self.metrics.right_ear = self.calculate_ear(right_eye)
            self.metrics.avg_ear = (self.metrics.left_ear + self.metrics.right_ear) / 2.0
            
            # Blink detection
            if self.metrics.avg_ear < self.EAR_BLINK_THRESHOLD:
                if not self.metrics.is_blinking:
                    self.metrics.is_blinking = True
                    self.metrics.total_blinks += 1
                    self.blink_counter += 1
                    self.last_blink_time = time.time()
            else:
                self.metrics.is_blinking = False
            
            # Calculate blink rate
            elapsed_time = time.time() - self.session_start_time
            if elapsed_time > 0:
                self.metrics.blink_rate = (self.metrics.total_blinks / elapsed_time) * 60
            
            # Calculate Mouth Aspect Ratio
            mouth = np.array([landmarks[i][:2] for i in self.MOUTH_INDICES])
            self.metrics.mar = self.calculate_mar(mouth)
            self.metrics.is_yawning = self.metrics.mar > self.MAR_YAWN_THRESHOLD
            
            # Calculate gaze direction (using left eye iris)
            left_iris = np.array([landmarks[i][:2] for i in self.LEFT_IRIS_INDICES])
            left_iris_center = np.mean(left_iris, axis=0)
            left_eye_corners = (landmarks[self.LEFT_EYE_INDICES[0]][:2], 
                              landmarks[self.LEFT_EYE_INDICES[3]][:2])
            
            self.metrics.gaze_horizontal, self.metrics.gaze_vertical = \
                self.calculate_gaze(left_iris_center, left_eye_corners)
            
            # Check if looking at screen
            self.metrics.looking_at_screen = abs(self.metrics.gaze_horizontal) < self.GAZE_THRESHOLD
            
            # Calculate head pose
            self.metrics.head_horizontal, self.metrics.head_vertical = \
                self.calculate_head_pose(landmarks, frame.shape)
            
            # Check if head is centered
            self.metrics.head_centered = (abs(self.metrics.head_horizontal) < self.HEAD_POSE_THRESHOLD and
                                         abs(self.metrics.head_vertical) < self.HEAD_POSE_THRESHOLD)
            
            # Calculate focus score
            self.metrics.focus_score = self.calculate_focus_score()
            
            # Draw face mesh on frame
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
        else:
            # No face detected
            self.metrics.focus_score = 0
            self.metrics.looking_at_screen = False
        
        return frame, self.metrics
    
    def draw_dashboard(self, frame: np.ndarray, metrics: FocusMetrics, 
                       session_focused_time: float, session_distracted_time: float,
                       pomodoro_time: Optional[int] = None) -> np.ndarray:
        """
        Draw real-time dashboard overlay on video frame.
        
        Args:
            frame: Input video frame
            metrics: Current focus metrics
            session_focused_time: Total focused time in seconds
            session_distracted_time: Total distracted time in seconds
            pomodoro_time: Remaining pomodoro time in seconds (optional)
            
        Returns:
            Frame with dashboard overlay
        """
        h, w, _ = frame.shape
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Dashboard background
        cv2.rectangle(overlay, (10, 10), (400, 350), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Title
        cv2.putText(frame, "Focus Detection Dashboard", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Focus Score (with color coding)
        score_color = (0, 255, 0) if metrics.focus_score >= 70 else \
                     (0, 255, 255) if metrics.focus_score >= 40 else (0, 0, 255)
        cv2.putText(frame, f"Focus Score: {metrics.focus_score}", (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, score_color, 2)
        
        # Eye metrics
        cv2.putText(frame, f"Avg EAR: {metrics.avg_ear:.3f}", (20, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Blinks: {metrics.total_blinks} ({metrics.blink_rate:.1f}/min)", 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Mouth metrics
        yawn_color = (0, 0, 255) if metrics.is_yawning else (255, 255, 255)
        cv2.putText(frame, f"MAR: {metrics.mar:.3f} {'[YAWN]' if metrics.is_yawning else ''}", 
                   (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, yawn_color, 1)
        
        # Gaze direction
        gaze_status = "Center" if metrics.looking_at_screen else "Away"
        gaze_color = (0, 255, 0) if metrics.looking_at_screen else (0, 0, 255)
        cv2.putText(frame, f"Gaze: {gaze_status}", (20, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, gaze_color, 1)
        
        # Head pose
        head_status = "Centered" if metrics.head_centered else "Off-center"
        head_color = (0, 255, 0) if metrics.head_centered else (0, 165, 255)
        cv2.putText(frame, f"Head: {head_status}", (20, 205),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, head_color, 1)
        
        # Session time
        total_time = session_focused_time + session_distracted_time
        cv2.putText(frame, f"Session Time: {int(total_time)}s", (20, 235),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Focused: {int(session_focused_time)}s", (20, 260),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"Distracted: {int(session_distracted_time)}s", (20, 285),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Pomodoro timer
        if pomodoro_time is not None:
            minutes = pomodoro_time // 60
            seconds = pomodoro_time % 60
            timer_text = f"Pomodoro: {minutes:02d}:{seconds:02d}"
            cv2.putText(frame, timer_text, (20, 315),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
        
        # Controls hint
        cv2.putText(frame, "Q:Quit | R:Reset | P:Pomodoro", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def reset_session(self):
        """Reset session statistics."""
        self.metrics.total_blinks = 0
        self.blink_counter = 0
        self.session_start_time = time.time()
        self.last_blink_time = time.time()
    
    def get_session_stats(self) -> dict:
        """
        Get current session statistics.
        
        Returns:
            Dictionary containing session statistics
        """
        elapsed_time = time.time() - self.session_start_time
        return {
            'total_blinks': self.metrics.total_blinks,
            'blink_rate': self.metrics.blink_rate,
            'session_duration': elapsed_time,
            'current_focus_score': self.metrics.focus_score
        }
