"""This is a helper class for viewing and anaylsing the data"""
import os

import numpy as np
import pandas as pd
import pickle

from . import yolo_helper
from Tennis_YOLO.tennis_utils import read_video
from Tennis_YOLO.tennis_utils import calculate_court_scale_from_keypoints

def video_meta_scrape(video_files):
    all_metadata = []
    for video_file in video_files:
        video_name = os.path.basename(video_file)

        print(f"Analyzing {video_name}...")

        try:
            # Read video to get basic info
            frames = read_video(video_file)

            if len(frames) > 0:
                metadata = {
                    'video_name': video_name,
                    'video_path': video_file,
                    'total_frames': len(frames),
                    'frame_shape': frames[0].shape,
                    'duration_seconds': len(frames) / 30.0,  # Assuming 30fps
                    'resolution': f"{frames[0].shape[1]}x{frames[0].shape[0]}",
                    'width': frames[0].shape[1],
                    'height': frames[0].shape[0],
                    'status': 'ready'
                }

                print(
                    f"  {metadata['total_frames']} frames, {metadata['resolution']}, {metadata['duration_seconds']:.1f}s")
            else:
                metadata = {
                    'video_name': video_name,
                    'video_path': video_file,
                    'total_frames': 0,
                    'status': 'error - no frames'
                }
                print(f"  Error: No frames loaded")

        except Exception as e:
            metadata = {
                'video_name': video_name,
                'video_path': video_file,
                'status': f'error - {str(e)}'
            }
            print(f"  Error: {str(e)}")

        all_metadata.append(metadata)

    return all_metadata


def process_single_video(video_metadata, use_live_detection=False):
    trackers_ready, player_tracker, ball_tracker, court_detector = yolo_helper.initialize_yolo_trackers()

    video_name = video_metadata['video_name']
    video_path = video_metadata['video_path']

    print(f"\nProcessing: {video_name}")
    print("-" * 50)

    results = {
        'video_name': video_name,
        'video_path': video_path,
        'processing_timestamp': pd.Timestamp.now(),
        'status': 'processing',
        'error_message': None
    }

    try:
        print("Loading video frames...")
        video_frames = read_video(video_path)

        if len(video_frames) == 0:
            raise ValueError("No frames loaded from video")

        print(f"Loaded {len(video_frames)} frames")

        # Create timestamps
        timestamps = pd.date_range(
            start='2024-01-01 00:00:00',
            periods=len(video_frames),
            freq='33.33ms'
        )

        # Detection processing
        if use_live_detection and trackers_ready:
            print("   ####->Running live YOLO detection...")

            # Player detection
            player_detections = player_tracker.detect_frames(video_frames)
            print(f"      Player detection complete: {len(player_detections)} frames")

            # Ball detection
            ball_detections = ball_tracker.detect_frames(video_frames)
            print(f"      Ball detection complete: {len(ball_detections)} frames")

            # Court detection
            court_keypoints = court_detector.predict(video_frames[0])
            print(f"      Court detection complete: {len(court_keypoints) // 2} keypoints")

        else:
            print("   Using stub detection data...")

            # Try to load pre-processed detection data
            try:
                with open('tracker_stubs/player_detections.pkl', 'rb') as f:
                    player_detections = pickle.load(f)
                    # Adjust length to match video frames
                    if len(player_detections) != len(video_frames):
                        if len(player_detections) < len(video_frames):
                            player_detections = player_detections * (len(video_frames) // len(player_detections) + 1)
                        player_detections = player_detections[:len(video_frames)]

                with open('tracker_stubs/ball_detections.pkl', 'rb') as f:
                    ball_detections = pickle.load(f)
                    # Adjust length to match video frames
                    if len(ball_detections) != len(video_frames):
                        if len(ball_detections) < len(video_frames):
                            ball_detections = ball_detections * (len(video_frames) // len(ball_detections) + 1)
                        ball_detections = ball_detections[:len(video_frames)]

                print(f"Loaded stub data: {len(player_detections)} player frames, {len(ball_detections)} ball frames")

                # Generate court keypoints based on video dimensions
                court_keypoints = court_detector.predict(video_frames[0]) if court_detector else None

            except Exception as e:
                print(f"     Error loading stub data: {e}")
                # Generate minimal mock data
                player_detections = [{} for _ in range(len(video_frames))]
                ball_detections = [{} for _ in range(len(video_frames))]
                court_keypoints = None

        # 4. Extract ball temporal data
        print("  ###### -> Processing ball tracking data...")
        ball_positions = [x.get(1, []) for x in ball_detections]

        ball_data = []
        for i, detection in enumerate(ball_positions):
            timestamp = timestamps[i]
            if len(detection) == 4:  # Valid detection
                ball_data.append({
                    'timestamp': timestamp,
                    'frame_number': i,
                    'video_name': video_name,
                    'x1': detection[0], 'y1': detection[1],
                    'x2': detection[2], 'y2': detection[3],
                    'center_x': (detection[0] + detection[2]) / 2,
                    'center_y': (detection[1] + detection[3]) / 2
                })
            else:  # Missing detection
                ball_data.append({
                    'timestamp': timestamp,
                    'frame_number': i,
                    'video_name': video_name,
                    'x1': np.nan, 'y1': np.nan,
                    'x2': np.nan, 'y2': np.nan,
                    'center_x': np.nan, 'center_y': np.nan
                })

        df_ball = pd.DataFrame(ball_data)
        ball_detection_rate = (~df_ball['center_x'].isna()).mean()
        print(f"    Ball data: {len(df_ball)} frames, {ball_detection_rate:.1%} detection rate")

        # Extract player temporal data
        print("  Processing player tracking data...")
        player_data = []

        # Filter players if court detection is available
        if court_keypoints is not None and player_tracker:
            try:
                filtered_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)
                print(f"    Player filtering applied")
            except:
                filtered_detections = player_detections
                print(f"     Player filtering failed, using original detections")
        else:
            filtered_detections = player_detections

        for frame_idx, frame_detections in enumerate(filtered_detections):
            timestamp = timestamps[frame_idx]

            if frame_detections and isinstance(frame_detections, dict):
                for player_id, bbox in frame_detections.items():
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        player_data.append({
                            'timestamp': timestamp,
                            'frame_number': frame_idx,
                            'video_name': video_name,
                            'player_id': int(player_id),
                            'x1': float(x1), 'y1': float(y1),
                            'x2': float(x2), 'y2': float(y2),
                            'center_x': float((x1 + x2) / 2),
                            'center_y': float((y1 + y2) / 2),
                            'width': float(x2 - x1),
                            'height': float(y2 - y1)
                        })

        df_players = pd.DataFrame(player_data)
        if len(df_players) > 0:
            unique_players = sorted(df_players['player_id'].unique())
            player_detection_rate = len(df_players) / len(video_frames)
            print(f"    Player data: {len(df_players)} detections, {len(unique_players)} unique players")
        else:
            print(f"     No player data extracted")
            unique_players = []

       # Court analysis
        print("  Processing court measurements...")
        if court_keypoints is not None:
            court_measurements = court_detector.get_court_measurements()
            court_scale = calculate_court_scale_from_keypoints(court_keypoints)
            print(f"     Court analysis complete: {len(court_keypoints) // 2} keypoints")
        else:
            # Default court measurements
            court_measurements = {
                'single_line_width': 8.23, 'double_line_width': 10.97,
                'half_court_height': 11.88, 'service_line_width': 6.4,
                'double_alley_difference': 1.37, 'no_mans_land_height': 5.48
            }
            court_scale = {
                'pixels_per_meter_x': 100, 'pixels_per_meter_y': 100,
                'court_width_pixels': 1000, 'court_height_pixels': 2000,
                'court_width_meters': 10.97, 'court_height_meters': 23.76
            }
            print(f"     Using default court measurements")

        print("  Calculating movement metrics...")

        # Ball metrics
        if len(df_ball) > 0:
            df_ball = df_ball.set_index('timestamp')

            # Ball movement calculations
            df_ball['prev_x'] = df_ball['center_x'].shift(1)
            df_ball['prev_y'] = df_ball['center_y'].shift(1)
            df_ball['distance_pixels'] = np.sqrt(
                (df_ball['center_x'] - df_ball['prev_x']) ** 2 +
                (df_ball['center_y'] - df_ball['prev_y']) ** 2
            )
            df_ball['distance_meters'] = df_ball['distance_pixels'] / court_scale['pixels_per_meter_x']
            df_ball['speed_ms'] = df_ball['distance_meters'] * 30
            df_ball['speed_kmh'] = df_ball['speed_ms'] * 3.6

            ball_stats = {
                'detection_rate': ball_detection_rate,
                'avg_speed_kmh': df_ball['speed_kmh'].mean(),
                'max_speed_kmh': df_ball['speed_kmh'].max(),
                'total_distance_meters': df_ball['distance_meters'].sum()
            }
        else:
            ball_stats = {'detection_rate': 0, 'avg_speed_kmh': 0, 'max_speed_kmh': 0, 'total_distance_meters': 0}

        # Player metrics
        player_stats = {}
        if len(df_players) > 0:
            df_players = df_players.set_index('timestamp')

            for player_id in unique_players:
                player_subset = df_players[df_players['player_id'] == player_id].copy()

                if len(player_subset) > 1:
                    # Movement calculations
                    player_subset['prev_x'] = player_subset['center_x'].shift(1)
                    player_subset['prev_y'] = player_subset['center_y'].shift(1)
                    player_subset['distance_pixels'] = np.sqrt(
                        (player_subset['center_x'] - player_subset['prev_x']) ** 2 +
                        (player_subset['center_y'] - player_subset['prev_y']) ** 2
                    )
                    player_subset['distance_meters'] = player_subset['distance_pixels'] / court_scale[
                        'pixels_per_meter_x']
                    player_subset['speed_ms'] = player_subset['distance_meters'] * 30
                    player_subset['speed_kmh'] = player_subset['speed_ms'] * 3.6

                    player_stats[player_id] = {
                        'detection_count': len(player_subset),
                        'detection_rate': len(player_subset) / len(video_frames),
                        'avg_speed_kmh': player_subset['speed_kmh'].mean(),
                        'max_speed_kmh': player_subset['speed_kmh'].max(),
                        'total_distance_meters': player_subset['distance_meters'].sum()
                    }

        print(f"      Metrics calculated for {len(player_stats)} players")


        results.update({
            'status': 'completed',
            'processing_time': (pd.Timestamp.now() - results['processing_timestamp']).total_seconds(),
            'video_metadata': video_metadata,
            'frame_count': len(video_frames),
            'duration_seconds': len(video_frames) / 30.0,
            'timestamps': timestamps,
            'ball_data': df_ball,
            'player_data': df_players,
            'court_measurements': court_measurements,
            'court_scale': court_scale,
            'ball_stats': ball_stats,
            'player_stats': player_stats,
            'unique_players': unique_players,
            'detection_method': 'live' if use_live_detection and trackers_ready else 'stub'
        })

        print(f" Processing completed in {results['processing_time']:.1f}s")

    except Exception as e:
        print(f"  Error processing video: {str(e)}")
        results.update({
            'status': 'error',
            'error_message': str(e),
            'processing_time': (pd.Timestamp.now() - results['processing_timestamp']).total_seconds()
        })

    return results