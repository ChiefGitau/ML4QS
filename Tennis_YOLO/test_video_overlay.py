#!/usr/bin/env python3
"""
Test script for tennis video overlay with MiniCourt functionality
"""

import cv2
import numpy as np
import pandas as pd
import os
from create_tennis_video_overlay import TennisVideoOverlayCreator
from tennis_utils import read_video

def test_video_overlay():

    video_files = [
        "input_videos/input_video.mp4",
        "input_videos/input_video_2.mp4",
        "input_videos/input_video_3.mp4"
    ]
    
    test_video = None
    for video_file in video_files:
        if os.path.exists(video_file):
            test_video = video_file
            break
    
    if not test_video:
        print("No test video found")
        return False
    
    print(f"Using test video: {test_video}")
    
    # Check if tracking data exists
    ball_csv = None
    player_csv = None
    
    if os.path.exists("ball_tracking.csv"):
        ball_csv = "ball_tracking.csv"
        print(f"Found ball tracking data: {ball_csv}")
    
    if os.path.exists("player_tracking.csv"):
        player_csv = "player_tracking.csv"
        print(f"Found player tracking data: {player_csv}")
    
    # Create output directory
    os.makedirs("test_output", exist_ok=True)

    print(f"\nTest 1: Creating basic video overlay...")
    
    try:
        overlay_creator = TennisVideoOverlayCreator()
        
        # Generate test output path
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"test_output/test_overlay_{timestamp}.mp4"
        
        # Load tracking data if available
        ball_df = None
        player_df = None
        
        if ball_csv:
            ball_df = pd.read_csv(ball_csv)
            print(f"   Loaded {len(ball_df)} ball tracking records")
        
        if player_csv:
            player_df = pd.read_csv(player_csv)
            print(f"   Loaded {len(player_df)} player tracking records")
        

        result_path = overlay_creator.create_video_with_overlay(
            video_path=test_video,
            ball_data_df=ball_df,
            player_data_df=player_df,
            output_path=output_path,
            start_frame=0,
            num_frames=30  # Just test first 30 frames
        )
        
        print(f" Video created at {result_path}")

        print(f"\n🔬 Verifying output video...")
        
        cap = cv2.VideoCapture(result_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Output video is readable")
                print(f" Frame shape: {frame.shape}")
                
                # Save a sample frame for inspection
                sample_frame_path = f"test_output/sample_frame_{timestamp}.jpg"
                cv2.imwrite(sample_frame_path, frame)
                print(f"Sample frame saved: {sample_frame_path}")
            else:
                print(f"Test 2 failed: Cannot read frame from output video")
            cap.release()
        else:
            print(f"Test 2 failed: Cannot open output video")
        

        if os.path.exists(result_path):
            file_size = os.path.getsize(result_path)
            print(f"Output file size: {file_size:,} bytes")
            if file_size < 1000:
                print(f"Warning")
        
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_minicourt_functionality():

    try:
        from tennis_utils import MiniCourt
        

        sample_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
        
        # Add some court-like features
        cv2.rectangle(sample_frame, (100, 100), (540, 380), (34, 139, 34), -1)  # Court green
        cv2.rectangle(sample_frame, (120, 120), (520, 360), (255, 255, 255), 2)  # Court lines
        cv2.line(sample_frame, (120, 240), (520, 240), (255, 255, 255), 3)     # Net

        
        # Initialize MiniCourt
        mini_court = MiniCourt(sample_frame)
        print("MiniCourt initialized")

        frame_with_court = mini_court.draw_background_rectangle(sample_frame.copy())
        frame_with_court = mini_court.draw_court(frame_with_court)
        print("Court drawing successful")

        test_pos = (320, 240)  # Center of frame
        mini_pos = mini_court.convert_position_to_mini_court(test_pos)
        print(f"Coordinate conversion: {test_pos} -> {mini_pos}")
        

        os.makedirs("test_output", exist_ok=True)
        cv2.imwrite("test_output/minicourt_test.jpg", frame_with_court)
        print("MiniCourt test image saved: test_output/minicourt_test.jpg")
        
        return True
        
    except Exception as e:
        print(f"MiniCourt test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":

    minicourt_ok = test_minicourt_functionality()

    overlay_ok = test_video_overlay()
    

    print(f"MiniCourt functionality: {'PASS' if minicourt_ok else ' FAIL'}")
    print(f"Video overlay creation: {'PASS' if overlay_ok else 'FAIL'}")
    
    if minicourt_ok and overlay_ok:
        print(f"\nPASSED!")
    else:
        print(f"\nSOME TESTS FAILED")
