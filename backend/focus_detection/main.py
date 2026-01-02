"""
Focus Detection Main Application

This module provides the main application loop for real-time focus detection
with webcam capture, distraction alerts, and Pomodoro timer functionality.
"""

import cv2
import time
import pygame
from typing import Optional
from detector import FocusDetector


class FocusApp:
    """
    Main Focus Detection Application.
    
    Features:
    - Webcam capture and real-time processing
    - Distraction alerts (sound + console)
    - Pomodoro timer (25 min focus, 5 min break)
    - Session time tracking
    - Keyboard controls
    - Session summary on exit
    """
    
    # Pomodoro settings
    POMODORO_FOCUS_TIME = 25 * 60  # 25 minutes in seconds
    POMODORO_BREAK_TIME = 5 * 60   # 5 minutes in seconds
    
    # Alert settings
    DISTRACTION_ALERT_THRESHOLD = 5  # seconds of being unfocused before alert
    FOCUS_SCORE_THRESHOLD = 50  # Below this score is considered unfocused
    
    def __init__(self):
        """Initialize the Focus Detection Application."""
        self.detector = FocusDetector()
        self.cap = None
        
        # Session tracking
        self.session_start_time = time.time()
        self.session_focused_time = 0.0
        self.session_distracted_time = 0.0
        self.last_update_time = time.time()
        
        # Distraction alert tracking
        self.unfocused_start_time = None
        self.alert_active = False
        
        # Pomodoro timer
        self.pomodoro_active = False
        self.pomodoro_start_time = None
        self.pomodoro_is_break = False
        
        # Initialize pygame for sound alerts
        pygame.mixer.init()
        self.alert_played = False
        
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """
        Initialize webcam capture.
        
        Args:
            camera_index: Index of the camera to use (default: 0)
            
        Returns:
            True if camera initialized successfully, False otherwise
        """
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully")
        return True
    
    def play_alert_sound(self):
        """Play distraction alert sound."""
        # Create a simple beep sound using pygame
        # Note: In production, you would load an actual sound file
        try:
            # Generate a beep tone (440 Hz for 0.5 seconds)
            sample_rate = 22050
            duration = 0.5
            frequency = 440
            
            # Generate samples
            n_samples = int(round(duration * sample_rate))
            buf = []
            for i in range(n_samples):
                value = int(4096 * ((i // (sample_rate // frequency)) % 2))
                buf.append(value)
            
            # Create sound from buffer
            sound = pygame.sndarray.make_sound(buf)
            sound.play()
            self.alert_played = True
        except Exception as e:
            # Fallback to console beep if pygame sound fails
            print(f"\a")  # ASCII bell character
            self.alert_played = True
    
    def check_distraction_alert(self, focus_score: int):
        """
        Check if user is distracted and trigger alerts.
        
        Args:
            focus_score: Current focus score (0-100)
        """
        current_time = time.time()
        
        if focus_score < self.FOCUS_SCORE_THRESHOLD:
            # User is unfocused
            if self.unfocused_start_time is None:
                self.unfocused_start_time = current_time
            
            unfocused_duration = current_time - self.unfocused_start_time
            
            if unfocused_duration >= self.DISTRACTION_ALERT_THRESHOLD and not self.alert_active:
                # Trigger alert
                self.alert_active = True
                print("\n" + "="*50)
                print("⚠️  DISTRACTION ALERT!")
                print(f"You've been unfocused for {int(unfocused_duration)} seconds")
                print("Please refocus on your studies!")
                print("="*50 + "\n")
                
                # Play sound alert
                if not self.alert_played:
                    self.play_alert_sound()
        else:
            # User is focused
            self.unfocused_start_time = None
            self.alert_active = False
            self.alert_played = False
    
    def update_session_time(self, focus_score: int):
        """
        Update session time tracking.
        
        Args:
            focus_score: Current focus score (0-100)
        """
        current_time = time.time()
        elapsed = current_time - self.last_update_time
        
        if focus_score >= self.FOCUS_SCORE_THRESHOLD:
            self.session_focused_time += elapsed
        else:
            self.session_distracted_time += elapsed
        
        self.last_update_time = current_time
    
    def start_pomodoro(self):
        """Start or toggle Pomodoro timer."""
        if not self.pomodoro_active:
            self.pomodoro_active = True
            self.pomodoro_start_time = time.time()
            self.pomodoro_is_break = False
            print("\n🍅 Pomodoro started! Focus for 25 minutes.")
        else:
            self.pomodoro_active = False
            self.pomodoro_start_time = None
            print("\n⏸️  Pomodoro paused.")
    
    def update_pomodoro(self):
        """Update Pomodoro timer and handle transitions."""
        if not self.pomodoro_active or self.pomodoro_start_time is None:
            return None
        
        elapsed = time.time() - self.pomodoro_start_time
        
        if not self.pomodoro_is_break:
            # Focus period
            remaining = self.POMODORO_FOCUS_TIME - int(elapsed)
            if remaining <= 0:
                # Focus period ended, start break
                print("\n✅ Focus session complete! Time for a 5-minute break.")
                self.pomodoro_is_break = True
                self.pomodoro_start_time = time.time()
                return self.POMODORO_BREAK_TIME
        else:
            # Break period
            remaining = self.POMODORO_BREAK_TIME - int(elapsed)
            if remaining <= 0:
                # Break ended
                print("\n⏰ Break time over! Ready for another focus session?")
                self.pomodoro_active = False
                self.pomodoro_start_time = None
                return None
        
        return max(0, remaining)
    
    def reset_session(self):
        """Reset session statistics."""
        self.session_start_time = time.time()
        self.session_focused_time = 0.0
        self.session_distracted_time = 0.0
        self.last_update_time = time.time()
        self.detector.reset_session()
        print("\n🔄 Session reset!")
    
    def print_session_summary(self):
        """Print session summary on exit."""
        total_time = self.session_focused_time + self.session_distracted_time
        focus_percentage = (self.session_focused_time / total_time * 100) if total_time > 0 else 0
        
        print("\n" + "="*60)
        print("📊 SESSION SUMMARY")
        print("="*60)
        print(f"Total Session Time: {int(total_time)} seconds ({total_time/60:.1f} minutes)")
        print(f"Focused Time: {int(self.session_focused_time)} seconds ({self.session_focused_time/60:.1f} minutes)")
        print(f"Distracted Time: {int(self.session_distracted_time)} seconds ({self.session_distracted_time/60:.1f} minutes)")
        print(f"Focus Percentage: {focus_percentage:.1f}%")
        
        stats = self.detector.get_session_stats()
        print(f"\nTotal Blinks: {stats['total_blinks']}")
        print(f"Average Blink Rate: {stats['blink_rate']:.1f} blinks/minute")
        
        # Provide feedback
        if focus_percentage >= 80:
            print("\n🌟 Excellent focus! Keep up the great work!")
        elif focus_percentage >= 60:
            print("\n👍 Good focus session. Room for improvement!")
        elif focus_percentage >= 40:
            print("\n⚠️  Moderate focus. Try to minimize distractions.")
        else:
            print("\n❌ Low focus session. Consider taking a break or changing environment.")
        
        print("="*60 + "\n")
    
    def run(self):
        """Main application loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*60)
        print("🎯 FOCUS DETECTION SYSTEM STARTED")
        print("="*60)
        print("\nControls:")
        print("  Q - Quit application")
        print("  R - Reset session statistics")
        print("  P - Start/pause Pomodoro timer")
        print("\nPress any key in the video window to begin...\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to capture frame")
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame for focus detection
                processed_frame, metrics = self.detector.process_frame(frame)
                
                # Update session time
                self.update_session_time(metrics.focus_score)
                
                # Check for distraction alerts
                self.check_distraction_alert(metrics.focus_score)
                
                # Update Pomodoro timer
                pomodoro_remaining = self.update_pomodoro()
                
                # Draw dashboard
                dashboard_frame = self.detector.draw_dashboard(
                    processed_frame,
                    metrics,
                    self.session_focused_time,
                    self.session_distracted_time,
                    pomodoro_remaining
                )
                
                # Display frame
                cv2.imshow('Focus Detection - Smart Study Planner', dashboard_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\nQuitting application...")
                    break
                elif key == ord('r') or key == ord('R'):
                    self.reset_session()
                elif key == ord('p') or key == ord('P'):
                    self.start_pomodoro()
                    
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            # Cleanup
            self.print_session_summary()
            
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()
            pygame.mixer.quit()
            
            print("Application closed successfully.")


def main():
    """Entry point for the focus detection application."""
    app = FocusApp()
    app.run()


if __name__ == "__main__":
    main()
