#!/usr/bin/env python3
"""
Production-Grade Elderly Monitoring AI System
Real-Time Human Activity, Posture & Fall Detection
Tech Stack: Python 3, OpenCV, MediaPipe Pose, NumPy

Author: Senior Computer Vision Engineer
Date: 2024
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional
from enum import Enum


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SystemConfig:
    """Central configuration for posture, activity, and fall detection."""
    
    # Camera & Display
    CAMERA_ID: int = 0
    DISPLAY_WIDTH: int = 1280
    DISPLAY_HEIGHT: int = 720
    TARGET_FPS: int = 30
    
    # MediaPipe Pose
    POSE_CONFIDENCE: float = 0.5
    POSE_TRACKING_CONFIDENCE: float = 0.5
    STATIC_IMAGE_MODE: bool = False
    
    # Temporal buffer (frame history)
    HISTORY_BUFFER_SIZE: int = 30
    MOTION_HISTORY_SIZE: int = 10
    
    # Posture Detection Thresholds
    TORSO_VERTICAL_THRESHOLD: float = 20.0  # degrees
    STANDING_HIP_HEIGHT_MIN: float = 0.65  # normalized (0-1)
    SITTING_HIP_HEIGHT_MAX: float = 0.50
    SITTING_KNEE_ANGLE_MIN: float = 70.0
    SITTING_KNEE_ANGLE_MAX: float = 140.0
    KNEE_EXTENDED_THRESHOLD: float = 160.0
    LYING_TORSO_ANGLE_THRESHOLD: float = 70.0
    BENDING_TORSO_ANGLE_MIN: float = 30.0
    BENDING_TORSO_ANGLE_MAX: float = 70.0
    
    # Activity Detection Thresholds
    MOVING_COM_VELOCITY_THRESHOLD: float = 0.03  # normalized pixels/frame
    MOVING_FRAME_DURATION: int = 5  # frames
    IDLE_THRESHOLD_DURATION: float = 10.0  # seconds
    IDLE_DISPLACEMENT_MAX: float = 0.02  # normalized
    
    # Fall Detection Thresholds
    FALL_HIP_VELOCITY_THRESHOLD: float = 0.15  # rapid descent
    FALL_IMPACT_DISPLACEMENT: float = 0.20  # large displacement
    FALL_LYING_DURATION: float = 0.5  # seconds in lying position
    FALL_DETECTION_WINDOW: float = 2.0  # seconds (stage 1->3)
    
    # Alerts
    INACTIVITY_ALERT_THRESHOLD: float = 60.0  # seconds


class PostureState(Enum):
    """Enumeration of posture states."""
    UNKNOWN = "Unknown"
    STANDING = "Standing"
    SITTING = "Sitting"
    LYING = "Lying"
    BENDING = "Bending"


class ActivityState(Enum):
    """Enumeration of activity states."""
    MOVING = "Moving"
    IDLE = "Idle"
    UNKNOWN = "Unknown"


class SafetyEvent(Enum):
    """Enumeration of safety events."""
    FALL = "FALL_DETECTED"
    LONG_INACTIVITY = "INACTIVITY_ALERT"
    NONE = "NONE"


# ============================================================================
# LOGGING SYSTEM
# ============================================================================

def setup_logging() -> logging.Logger:
    """Configure structured logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ============================================================================
# UTILITY CLASSES
# ============================================================================

@dataclass
class FPSCounter:
    """Efficient FPS counter with rolling average."""
    
    window_size: int = 30
    fps_history: deque = field(default_factory=lambda: deque(maxlen=30))
    last_time: float = field(default_factory=time.time)
    
    def update(self) -> float:
        """Update FPS counter and return current FPS."""
        current_time = time.time()
        delta = current_time - self.last_time
        
        if delta > 0:
            current_fps = 1.0 / delta
            self.fps_history.append(current_fps)
        
        self.last_time = current_time
        
        if self.fps_history:
            return np.mean(list(self.fps_history))
        return 0.0


@dataclass
class PoseLandmarks:
    """Container for pose landmarks extracted from MediaPipe."""
    
    landmarks: np.ndarray  # Shape: (33, 3) [x, y, z]
    visibility: np.ndarray  # Shape: (33,)
    height: int
    width: int
    
    def is_valid(self) -> bool:
        """Check if pose landmarks are valid."""
        return (self.landmarks is not None and 
                len(self.landmarks) == 33 and 
                np.mean(self.visibility) > 0.5)
    
    def normalize(self) -> np.ndarray:
        """Normalize landmarks to [0,1] range using image dimensions."""
        normalized = self.landmarks.copy()
        # Only normalize x,y coordinates (z is depth, keep as-is)
        normalized[:, 0] /= self.width
        normalized[:, 1] /= self.height
        # z coordinate stays as relative depth (already ~normalized by MediaPipe)
        return normalized


@dataclass
class MotionFrame:
    """Container for motion-related metrics from a single frame."""
    
    timestamp: float
    com_position: np.ndarray  # Center of mass [x, y]
    shoulder_pos: np.ndarray
    hip_pos: np.ndarray
    landmarks_normalized: np.ndarray
    
    def displacement_to(self, other: 'MotionFrame') -> float:
        """Compute Euclidean distance to another frame's COM."""
        if other is None:
            return 0.0
        return float(np.linalg.norm(self.com_position - other.com_position))


@dataclass
class PoseMetrics:
    """Container for computed pose geometry and joint angles."""
    
    body_height: float
    shoulder_width: float
    hip_width: float
    torso_angle: float  # degrees, 0=vertical, 90=horizontal
    hip_height: float  # normalized (0-1)
    shoulder_height: float
    left_knee_angle: float  # degrees
    right_knee_angle: float
    left_hip_angle: float
    right_hip_angle: float
    center_of_mass: np.ndarray  # 2D [x, y]


@dataclass
class SystemState:
    """Current system state snapshot."""
    
    posture: PostureState = PostureState.UNKNOWN
    activity: ActivityState = ActivityState.UNKNOWN
    safety_event: SafetyEvent = SafetyEvent.NONE
    pose_metrics: Optional[PoseMetrics] = None
    com_velocity: float = 0.0
    shoulder_velocity: float = 0.0
    hip_velocity: float = 0.0
    inactivity_duration: float = 0.0
    frame_number: int = 0
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# POSE ANALYZER
# ============================================================================

class PoseAnalyzer:
    """
    Extracts and computes pose geometry from MediaPipe landmarks.
    Uses vector math for joint angles, body dimensions, and orientation.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logger
    
    def analyze(self, landmarks: PoseLandmarks) -> PoseMetrics:
        """
        Compute all pose metrics from normalized landmarks.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            PoseMetrics with computed geometry
        """
        if not landmarks.is_valid():
            return self._null_metrics()
        
        normalized = landmarks.normalize()
        
        # Extract key body points
        nose = normalized[0]
        left_shoulder = normalized[11]
        right_shoulder = normalized[12]
        left_hip = normalized[23]
        right_hip = normalized[24]
        left_ankle = normalized[27]
        right_ankle = normalized[28]
        left_knee = normalized[25]
        right_knee = normalized[26]
        
        # Compute center of mass (average of hips) - use only x,y (2D)
        com = (left_hip[:2] + right_hip[:2]) / 2.0
        
        # Body dimensions
        avg_ankle = (left_ankle + right_ankle) / 2.0
        body_height = float(np.linalg.norm(nose - avg_ankle))
        
        shoulder_width = float(np.linalg.norm(right_shoulder - left_shoulder))
        hip_width = float(np.linalg.norm(right_hip - left_hip))
        
        # Torso angle: angle between vertical and shoulder→hip vector
        shoulder_mid = (left_shoulder + right_shoulder) / 2.0
        hip_mid = (left_hip + right_hip) / 2.0
        torso_vector = hip_mid - shoulder_mid
        torso_angle = self._angle_from_vertical(torso_vector)
        
        # Heights
        hip_height = float(hip_mid[1])  # y-coordinate normalized
        shoulder_height = float(shoulder_mid[1])
        
        # Joint angles (knee and hip)
        left_knee_angle = self._angle_between_vectors(
            left_hip - left_knee,
            left_ankle - left_knee
        )
        right_knee_angle = self._angle_between_vectors(
            right_hip - right_knee,
            right_ankle - right_knee
        )
        
        left_hip_angle = self._angle_between_vectors(
            left_shoulder - left_hip,
            left_knee - left_hip
        )
        right_hip_angle = self._angle_between_vectors(
            right_shoulder - right_hip,
            right_knee - right_hip
        )
        
        metrics = PoseMetrics(
            body_height=body_height,
            shoulder_width=shoulder_width,
            hip_width=hip_width,
            torso_angle=torso_angle,
            hip_height=hip_height,
            shoulder_height=shoulder_height,
            left_knee_angle=left_knee_angle,
            right_knee_angle=right_knee_angle,
            left_hip_angle=left_hip_angle,
            right_hip_angle=right_hip_angle,
            center_of_mass=com
        )
        
        return metrics
    
    @staticmethod
    def _angle_from_vertical(vector: np.ndarray) -> float:
        """
        Compute angle from vertical (upward direction).
        0° = vertical, 90° = horizontal.
        Uses only x,y coordinates (2D projection).
        """
        vertical = np.array([0.0, -1.0])  # upward direction
        vector_2d = vector[:2] if len(vector) >= 2 else vector
        cos_angle = np.dot(vector_2d, vertical) / (np.linalg.norm(vector_2d) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return float(np.degrees(angle_rad))
    
    @staticmethod
    def _angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute angle between two vectors in degrees using 2D projections."""
        # Use only x,y coordinates
        v1_2d = v1[:2] if len(v1) >= 2 else v1
        v2_2d = v2[:2] if len(v2) >= 2 else v2
        
        norm1 = np.linalg.norm(v1_2d)
        norm2 = np.linalg.norm(v2_2d)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(v1_2d, v2_2d) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        return float(np.degrees(angle_rad))
    
    @staticmethod
    def _null_metrics() -> PoseMetrics:
        """Return null/default metrics."""
        return PoseMetrics(
            body_height=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
            torso_angle=0.0,
            hip_height=0.0,
            shoulder_height=0.0,
            left_knee_angle=0.0,
            right_knee_angle=0.0,
            left_hip_angle=0.0,
            right_hip_angle=0.0,
            center_of_mass=np.array([0.0, 0.0])  # 2D
        )


# ============================================================================
# MOTION ANALYZER
# ============================================================================

class MotionAnalyzer:
    """
    Tracks motion metrics over time using frame history buffer.
    Computes velocities and displacements for activity classification.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.history: deque = deque(maxlen=config.HISTORY_BUFFER_SIZE)
        self.logger = logger
    
    def update(self, motion_frame: MotionFrame) -> None:
        """Add new motion frame to history buffer."""
        self.history.append(motion_frame)
    
    def get_com_velocity(self) -> float:
        """
        Compute center-of-mass velocity (pixels/frame).
        Uses last 2 frames for instantaneous velocity.
        """
        if len(self.history) < 2:
            return 0.0
        
        current = self.history[-1]
        previous = self.history[-2]
        
        displacement = current.displacement_to(previous)
        time_delta = current.timestamp - previous.timestamp
        
        if time_delta <= 0:
            return 0.0
        
        velocity = displacement / time_delta
        return float(velocity)
    
    def get_shoulder_velocity(self) -> float:
        """Compute shoulder velocity."""
        if len(self.history) < 2:
            return 0.0
        
        current = self.history[-1]
        previous = self.history[-2]
        
        shoulder_displacement = np.linalg.norm(
            current.shoulder_pos - previous.shoulder_pos
        )
        time_delta = current.timestamp - previous.timestamp
        
        if time_delta <= 0:
            return 0.0
        
        return float(shoulder_displacement / time_delta)
    
    def get_hip_velocity(self) -> float:
        """Compute hip velocity (especially downward component)."""
        if len(self.history) < 2:
            return 0.0
        
        current = self.history[-1]
        previous = self.history[-2]
        
        hip_displacement = np.linalg.norm(
            current.hip_pos - previous.hip_pos
        )
        time_delta = current.timestamp - previous.timestamp
        
        if time_delta <= 0:
            return 0.0
        
        return float(hip_displacement / time_delta)
    
    def get_max_displacement_in_window(self, window_size: int = 10) -> float:
        """Compute max displacement within last N frames."""
        if len(self.history) < 2:
            return 0.0
        
        frames = list(self.history)[-window_size:]
        if not frames:
            return 0.0
        
        displacements = []
        for i in range(1, len(frames)):
            disp = frames[i].displacement_to(frames[i-1])
            displacements.append(disp)
        
        return float(max(displacements)) if displacements else 0.0
    
    def get_average_displacement(self, window_size: int = 30) -> float:
        """Compute average displacement over window."""
        if len(self.history) < 2:
            return 0.0
        
        frames = list(self.history)[-window_size:]
        displacements = []
        
        for i in range(1, len(frames)):
            disp = frames[i].displacement_to(frames[i-1])
            displacements.append(disp)
        
        return float(np.mean(displacements)) if displacements else 0.0


# ============================================================================
# POSTURE CLASSIFIER
# ============================================================================

class PostureClassifier:
    """
    Classifies posture using rule-based geometry thresholds.
    Outputs: Standing, Sitting, Lying, Bending.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logger
    
    def classify(self, metrics: PoseMetrics) -> PostureState:
        """
        Classify posture from pose metrics using geometric rules.
        Priority: Lying > Sitting > Bending > Standing
        """
        
        if metrics.body_height == 0.0:
            return PostureState.UNKNOWN
        
        avg_knee_angle = (metrics.left_knee_angle + metrics.right_knee_angle) / 2.0
        
        # Check Lying: torso very horizontal (>70°) AND shoulder-hip height diff very small
        # Lying person has shoulder and hip at nearly same y-coordinate
        if metrics.torso_angle > self.config.LYING_TORSO_ANGLE_THRESHOLD:
            height_diff = abs(metrics.shoulder_height - metrics.hip_height)
            # In true lying, shoulder ≈ hip height (diff < 0.08)
            if height_diff < 0.08:
                return PostureState.LYING
        
        # Check Sitting: hip drops significantly, knees bent, upright torso
        # Sitting: hip_height typically 0.30-0.50 (depending on chair height)
        if (metrics.hip_height < 0.55 and
            self.config.SITTING_KNEE_ANGLE_MIN <= avg_knee_angle <= self.config.SITTING_KNEE_ANGLE_MAX and
            metrics.torso_angle < self.config.TORSO_VERTICAL_THRESHOLD):
            return PostureState.SITTING
        
        # Check Bending: forward lean (30-70°), hip stays high but torso tilts
        if (self.config.BENDING_TORSO_ANGLE_MIN <= metrics.torso_angle <= 
            self.config.BENDING_TORSO_ANGLE_MAX and
            avg_knee_angle > 120.0 and
            metrics.hip_height > 0.50):
            return PostureState.BENDING
        
        # Check Standing: MAIN CASE
        # Standing: nearly vertical torso (<20°), knees extended (>160°), hip mid-height (0.40-0.70)
        if (metrics.torso_angle < self.config.TORSO_VERTICAL_THRESHOLD and
            avg_knee_angle > self.config.KNEE_EXTENDED_THRESHOLD and
            metrics.hip_height > 0.40):
            return PostureState.STANDING
        
        return PostureState.UNKNOWN


# ============================================================================
# ACTIVITY CLASSIFIER
# ============================================================================

class ActivityClassifier:
    """
    Classifies activity (Moving vs Idle) based on motion metrics.
    Uses temporal windows for robust classification.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.moving_frame_count: int = 0
        self.idle_start_time: float = time.time()
        self.last_activity_time: float = time.time()
        self.logger = logger
    
    def classify(self, motion: MotionAnalyzer) -> Tuple[ActivityState, float]:
        """
        Classify activity state.
        Returns: (ActivityState, inactivity_duration_seconds)
        """
        
        com_velocity = motion.get_com_velocity()
        current_time = time.time()
        
        # If COM moving fast, reset idle timer
        if com_velocity > self.config.MOVING_COM_VELOCITY_THRESHOLD:
            self.moving_frame_count += 1
            self.last_activity_time = current_time
        else:
            self.moving_frame_count = max(0, self.moving_frame_count - 1)
        
        # Determine activity state
        if self.moving_frame_count >= self.config.MOVING_FRAME_DURATION:
            activity = ActivityState.MOVING
        else:
            activity = ActivityState.IDLE
        
        # Compute inactivity duration
        inactivity_duration = current_time - self.last_activity_time
        
        return activity, inactivity_duration


# ============================================================================
# FALL DETECTOR
# ============================================================================

class FallDetector:
    """
    Multi-stage fall detection using rapid descent → impact → lying state.
    Temporal validation prevents false positives.
    """
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logger
        self.stage1_triggered = False
        self.stage1_time: float = 0.0
        self.fall_confirmed = False
        self.fall_time: float = 0.0
    
    def detect(self, motion: MotionAnalyzer, 
               posture: PostureState,
               pose_metrics: PoseMetrics) -> SafetyEvent:
        """
        Multi-stage fall detection pipeline.
        Stage 1: Rapid descent (large downward hip velocity)
        Stage 2: Large displacement (impact)
        Stage 3: Lying posture (rest state)
        """
        
        current_time = time.time()
        hip_velocity = motion.get_hip_velocity()
        max_displacement = motion.get_max_displacement_in_window(5)
        
        # STAGE 1: Rapid descent
        if hip_velocity > self.config.FALL_HIP_VELOCITY_THRESHOLD:
            if not self.stage1_triggered:
                self.stage1_triggered = True
                self.stage1_time = current_time
                self.logger.info(f"Fall Stage 1 triggered: hip_velocity={hip_velocity:.3f}")
        
        # Reset Stage 1 if timeout
        if (self.stage1_triggered and 
            current_time - self.stage1_time > self.config.FALL_DETECTION_WINDOW):
            self.stage1_triggered = False
        
        # STAGE 2 & 3: Large displacement + lying state
        if self.stage1_triggered:
            if (max_displacement > self.config.FALL_IMPACT_DISPLACEMENT and
                posture == PostureState.LYING):
                
                self.fall_confirmed = True
                self.fall_time = current_time
                self.logger.critical("FALL DETECTED: Multi-stage confirmation!")
                return SafetyEvent.FALL
        
        # Timeout fall confirmation
        if (self.fall_confirmed and 
            current_time - self.fall_time > 0.1):  # Reset quickly for re-detection
            self.fall_confirmed = False
            self.stage1_triggered = False
        
        return SafetyEvent.NONE


# ============================================================================
# ELDERLY MONITORING SYSTEM
# ============================================================================

class ElderlyMonitoringSystem:
    """
    Central system orchestrator.
    Integrates: Pose extraction → Analysis → Classification → Safety.
    Manages real-time video capture and visualization.
    """
    
    def __init__(self, config: SystemConfig = None):
        self.config = config or SystemConfig()
        self.logger = logger
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose.Pose(
            static_image_mode=self.config.STATIC_IMAGE_MODE,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=self.config.POSE_CONFIDENCE,
            min_tracking_confidence=self.config.POSE_TRACKING_CONFIDENCE
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize subsystems
        self.pose_analyzer = PoseAnalyzer(self.config)
        self.motion_analyzer = MotionAnalyzer(self.config)
        self.posture_classifier = PostureClassifier(self.config)
        self.activity_classifier = ActivityClassifier(self.config)
        self.fall_detector = FallDetector(self.config)
        self.fps_counter = FPSCounter(window_size=30)
        
        # State tracking
        self.current_state = SystemState()
        self.frame_count = 0
        self.system_start_time = time.time()
        
        self.logger.info("=" * 70)
        self.logger.info("Elderly Monitoring System Initialized")
        self.logger.info(f"Config: {self.config}")
        self.logger.info("=" * 70)
    
    def run(self) -> None:
        """Main real-time processing loop."""
        
        cap = cv2.VideoCapture(self.config.CAMERA_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.DISPLAY_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.DISPLAY_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.config.TARGET_FPS)
        
        if not cap.isOpened():
            self.logger.error("Failed to open camera!")
            return
        
        self.logger.info(f"Camera opened. Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.mp_pose.process(frame_rgb)
                
                # Extract and analyze pose
                self._process_frame(frame, results, frame_rgb.shape)
                
                # Update FPS
                fps = self.fps_counter.update()
                
                # Visualize
                self._render_visualization(frame, fps)
                
                # Display
                cv2.imshow('Elderly Monitoring System', frame)
                
                # Controls
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC
                    self.logger.info("Exiting...")
                    break
                elif key == ord('r'):  # Reset stats
                    self.logger.info("Resetting activity timer")
                    self.activity_classifier.last_activity_time = time.time()
                
                self.frame_count += 1
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info(f"System shutdown. Total frames: {self.frame_count}")
    
    def _process_frame(self, frame: np.ndarray, results, shape: Tuple) -> None:
        """Process single frame: pose extraction → analysis → classification."""
        
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Extract pose landmarks
        if results.pose_landmarks is None:
            self.current_state.posture = PostureState.UNKNOWN
            self.current_state.activity = ActivityState.UNKNOWN
            return
        
        # Convert MediaPipe landmarks to array
        landmarks_array = np.array([
            [lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark
        ])
        visibility_array = np.array([
            lm.visibility for lm in results.pose_landmarks.landmark
        ])
        
        pose_landmarks = PoseLandmarks(
            landmarks=landmarks_array,
            visibility=visibility_array,
            height=height,
            width=width
        )
        
        # Analyze pose geometry
        pose_metrics = self.pose_analyzer.analyze(pose_landmarks)
        
        # Create motion frame with 2D positions
        normalized = pose_landmarks.normalize()
        shoulder_mid = (normalized[11][:2] + normalized[12][:2]) / 2.0
        hip_mid = (normalized[23][:2] + normalized[24][:2]) / 2.0
        
        motion_frame = MotionFrame(
            timestamp=current_time,
            com_position=pose_metrics.center_of_mass,
            shoulder_pos=shoulder_mid,
            hip_pos=hip_mid,
            landmarks_normalized=normalized
        )
        
        # Update motion history
        self.motion_analyzer.update(motion_frame)
        
        # Classify posture
        posture = self.posture_classifier.classify(pose_metrics)
        
        # Classify activity
        activity, inactivity_duration = self.activity_classifier.classify(
            self.motion_analyzer
        )
        
        # Detect fall
        safety_event = self.fall_detector.detect(
            self.motion_analyzer, posture, pose_metrics
        )
        
        # Update system state
        self.current_state = SystemState(
            posture=posture,
            activity=activity,
            safety_event=safety_event,
            pose_metrics=pose_metrics,
            com_velocity=self.motion_analyzer.get_com_velocity(),
            shoulder_velocity=self.motion_analyzer.get_shoulder_velocity(),
            hip_velocity=self.motion_analyzer.get_hip_velocity(),
            inactivity_duration=inactivity_duration,
            frame_number=self.frame_count,
            timestamp=current_time
        )
        
        # Log state periodically
        if self.frame_count % 30 == 0:
            self._log_system_state()
    
    def _render_visualization(self, frame: np.ndarray, fps: float) -> None:
        """Render all visualization overlays on frame."""
        
        height, width = frame.shape[:2]
        
        # Draw pose skeleton if available
        if self.current_state.pose_metrics is not None:
            self._draw_skeleton(frame)
        
        # Draw metrics panel
        self._draw_metrics_panel(frame, fps)
        
        # Draw alerts
        if self.current_state.safety_event == SafetyEvent.FALL:
            self._draw_fall_alert(frame)
        
        if self.current_state.inactivity_duration > self.config.INACTIVITY_ALERT_THRESHOLD:
            self._draw_inactivity_alert(frame)
    
    def _draw_skeleton(self, frame: np.ndarray) -> None:
        """Draw pose skeleton on frame."""
        
        height, width = frame.shape[:2]
        
        if self.motion_analyzer.history:
            motion_frame = self.motion_analyzer.history[-1]
            normalized = motion_frame.landmarks_normalized
            landmarks_pixels = normalized.copy()
            landmarks_pixels[:, 0] *= width
            landmarks_pixels[:, 1] *= height
            
            # Draw lines connecting joints
            connections = [
                (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                (11, 23), (12, 24), (23, 24), (23, 25), (25, 27),
                (24, 26), (26, 28)
            ]
            
            for start, end in connections:
                pt1 = tuple(landmarks_pixels[start, :2].astype(int))
                pt2 = tuple(landmarks_pixels[end, :2].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            # Draw joints
            for i, lm in enumerate(landmarks_pixels):
                pt = tuple(lm[:2].astype(int))
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
    
    def _draw_metrics_panel(self, frame: np.ndarray, fps: float) -> None:
        """Draw metrics panel with posture, activity, velocities."""
        
        height, width = frame.shape[:2]
        
        # Panel background
        panel_height = 200
        cv2.rectangle(frame, (0, 0), (350, panel_height), (0, 0, 0), -1)
        cv2.rectangle(frame, (0, 0), (350, panel_height), (0, 255, 0), 2)
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        color_text = (255, 255, 255)
        
        y_offset = 25
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, y_offset), font, font_scale, 
                   color_text, thickness)
        y_offset += 25
        
        # Posture
        posture_text = f"Posture: {self.current_state.posture.value}"
        cv2.putText(frame, posture_text, (10, y_offset), font, font_scale, 
                   color_text, thickness)
        y_offset += 25
        
        # Activity
        activity_text = f"Activity: {self.current_state.activity.value}"
        cv2.putText(frame, activity_text, (10, y_offset), font, font_scale, 
                   color_text, thickness)
        y_offset += 25
        
        # COM Velocity
        com_vel_text = f"COM Vel: {self.current_state.com_velocity:.3f} px/s"
        cv2.putText(frame, com_vel_text, (10, y_offset), font, font_scale, 
                   color_text, thickness)
        y_offset += 25
        
        # Hip Velocity
        hip_vel_text = f"Hip Vel: {self.current_state.hip_velocity:.3f} px/s"
        cv2.putText(frame, hip_vel_text, (10, y_offset), font, font_scale, 
                   color_text, thickness)
        y_offset += 25
        
        # Inactivity
        inactivity_text = f"Inactive: {self.current_state.inactivity_duration:.1f}s"
        cv2.putText(frame, inactivity_text, (10, y_offset), font, font_scale, 
                   color_text, thickness)
        y_offset += 25
        
        # Frame count
        cv2.putText(frame, f"Frame: {self.frame_count}", (10, y_offset), font, 
                   font_scale, color_text, thickness)
    
    def _draw_fall_alert(self, frame: np.ndarray) -> None:
        """Draw flashing FALL ALERT."""
        
        height, width = frame.shape[:2]
        
        # Red overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        # Alert text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "!!! FALL DETECTED !!!"
        font_scale = 2.0
        thickness = 3
        color = (0, 0, 255)
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height // 2) + (text_size[1] // 2)
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
        
        self.logger.critical("FALL ALERT DISPLAYED")
    
    def _draw_inactivity_alert(self, frame: np.ndarray) -> None:
        """Draw inactivity warning."""
        
        height, width = frame.shape[:2]
        
        # Yellow border
        cv2.rectangle(frame, (0, 0), (width, height), (0, 255, 255), 5)
        
        # Warning text
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = "INACTIVITY ALERT"
        font_scale = 1.5
        thickness = 2
        color = (0, 255, 255)
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 30
        
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)
    
    def _log_system_state(self) -> None:
        """Log system state periodically."""
        
        elapsed = time.time() - self.system_start_time
        
        log_msg = (
            f"[Frame {self.frame_count} @ {elapsed:.1f}s] "
            f"Posture={self.current_state.posture.value}, "
            f"Activity={self.current_state.activity.value}, "
            f"COM_Vel={self.current_state.com_velocity:.3f}, "
            f"Hip_Vel={self.current_state.hip_velocity:.3f}, "
            f"Inactive={self.current_state.inactivity_duration:.1f}s"
        )
        self.logger.info(log_msg)
        
        # Debug metrics
        if self.current_state.pose_metrics:
            metrics = self.current_state.pose_metrics
            avg_knee = (metrics.left_knee_angle + metrics.right_knee_angle) / 2.0
            debug_msg = (
                f"  [DEBUG] torso_angle={metrics.torso_angle:.1f}° "
                f"hip_height={metrics.hip_height:.3f} "
                f"shoulder_height={metrics.shoulder_height:.3f} "
                f"knee_angle={avg_knee:.1f}° "
                f"height_diff={abs(metrics.shoulder_height - metrics.hip_height):.3f}"
            )
            self.logger.info(debug_msg)
        
        if self.current_state.safety_event != SafetyEvent.NONE:
            self.logger.warning(f"SAFETY EVENT: {self.current_state.safety_event.value}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Initialize and run the monitoring system."""
    
    # Create system with custom or default config
    config = SystemConfig()
    system = ElderlyMonitoringSystem(config)
    
    # Run real-time processing
    system.run()


if __name__ == "__main__":
    main()