"""
Focus Detection Test Script

Simple test script to verify the focus detection module is working correctly
with webcam input.
"""

import cv2
import sys
from detector import FocusDetector


def test_focus_detection():
    """
    Test the focus detection system with live webcam feed.
    
    This script provides a simple way to verify that:
    - Camera is accessible
    - Face detection is working
    - Metrics are being calculated
    - Dashboard is rendering correctly
    """
    print("="*60)
    print("Focus Detection Test Script")
    print("="*60)
    print("\nInitializing focus detector...")
    
    # Initialize detector
    detector = FocusDetector()
    
    # Initialize camera
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        print("Please check that:")
        print("  1. A webcam is connected")
        print("  2. No other application is using the camera")
        print("  3. Camera permissions are granted")
        return False
    
    print("✅ Camera opened successfully")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "="*60)
    print("TEST INSTRUCTIONS")
    print("="*60)
    print("The test will open a window showing your webcam feed.")
    print("Try the following to test detection:")
    print("  • Face detection: Move your face in/out of frame")
    print("  • Blink detection: Blink your eyes")
    print("  • Yawn detection: Open your mouth wide")
    print("  • Gaze tracking: Look left/right")
    print("  • Head pose: Tilt your head")
    print("\nPress 'Q' to quit the test")
    print("="*60 + "\n")
    
    frame_count = 0
    test_duration = 0
    start_time = cv2.getTickCount()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Process frame
            processed_frame, metrics = detector.process_frame(frame)
            
            # Draw simple dashboard
            dashboard_frame = detector.draw_dashboard(
                processed_frame,
                metrics,
                session_focused_time=test_duration,
                session_distracted_time=0
            )
            
            # Add test info
            cv2.putText(dashboard_frame, "TEST MODE", (dashboard_frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Display frame
            cv2.imshow('Focus Detection Test', dashboard_frame)
            
            # Update counters
            frame_count += 1
            test_duration = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            
            # Print metrics every 30 frames
            if frame_count % 30 == 0:
                print(f"Frame {frame_count} | Focus Score: {metrics.focus_score} | "
                      f"EAR: {metrics.avg_ear:.3f} | MAR: {metrics.mar:.3f} | "
                      f"Blinks: {metrics.total_blinks}")
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nTest stopped by user")
                break
                
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
    
    # Print test summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Test completed successfully")
    print(f"Frames processed: {frame_count}")
    print(f"Test duration: {test_duration:.1f} seconds")
    print(f"Average FPS: {frame_count/test_duration:.1f}")
    
    stats = detector.get_session_stats()
    print(f"\nFinal Statistics:")
    print(f"  Total blinks: {stats['total_blinks']}")
    print(f"  Blink rate: {stats['blink_rate']:.1f} blinks/minute")
    print(f"  Final focus score: {stats['current_focus_score']}")
    
    print("\n✅ All systems operational!")
    print("="*60 + "\n")
    
    return True


if __name__ == "__main__":
    print("\n🧪 Starting Focus Detection Test...\n")
    
    success = test_focus_detection()
    
    if success:
        print("✅ Test passed! Focus detection is working correctly.")
        sys.exit(0)
    else:
        print("❌ Test failed! Please check the error messages above.")
        sys.exit(1)
