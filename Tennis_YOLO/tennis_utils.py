
import cv2
import pickle
import pandas as pd
import numpy as np
import os

def read_video(video_path):
    """
    Read video frames using OpenCV
    Alternative to original utils.read_video
    """
    frames = []
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"Loaded {len(frames)} frames from {video_path}")
    else:
        print(f"Video file not found: {video_path}")
        print("Creating dummy frames for demonstration...")
        # Create dummy frames for demonstration
        for i in range(300):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * (50 + i % 50)
            frames.append(frame)
        print(f"Created {len(frames)} dummy frames")
    return frames

def save_video(frames, output_path, fps=30):
    """Save video frames to file"""
    if len(frames) == 0:
        print("No frames to save")
        return False
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Saved {len(frames)} frames to {output_path}")
    return True

def measure_distance(p1, p2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def draw_player_stats(*args, **kwargs):
    """Mock function for drawing player stats"""
    print("Drawing player stats (mock implementation)")
    return None

def convert_pixel_distance_to_meters(pixel_distance, reference_height_meters=1.88, reference_height_pixels=100):
    """Convert pixel distance to meters using player height as reference"""
    if reference_height_pixels == 0:
        return pixel_distance * 0.01  # Fallback scale factor
    
    meters_per_pixel = reference_height_meters / reference_height_pixels
    return pixel_distance * meters_per_pixel

def convert_meters_to_pixel_distance(meters, reference_meters, reference_pixels):
    """Convert meters to pixel distance using a reference measurement"""
    if reference_meters == 0:
        return meters * 100  # Fallback
    return meters * (reference_pixels / reference_meters)

def get_closest_keypoint_index(position, keypoints, keypoint_indices=None):
    """Find the closest court keypoint to a given position"""
    if keypoint_indices is None:
        keypoint_indices = list(range(len(keypoints) // 2))
    
    points = keypoints.reshape(-1, 2)
    min_distance = float('inf')
    closest_index = 0
    
    for idx in keypoint_indices:
        if idx < len(points):
            distance = measure_distance(position, points[idx])
            if distance < min_distance:
                min_distance = distance
                closest_index = idx
                
    return closest_index

def calculate_court_scale_from_keypoints(keypoints):
    """Calculate scale factors from court keypoints"""
    points = keypoints.reshape(-1, 2)
    
    # Standard tennis court measurements
    DOUBLE_LINE_WIDTH = 10.97  # meters
    COURT_LENGTH = 23.76  # meters (11.88 * 2)
    
    # Calculate court dimensions in pixels
    if len(points) >= 4:
        court_width_pixels = abs(points[1][0] - points[0][0])  # Right - Left baseline
        court_height_pixels = abs(points[0][1] - points[2][1])  # Near - Far baseline
        
        pixels_per_meter_x = court_width_pixels / DOUBLE_LINE_WIDTH
        pixels_per_meter_y = court_height_pixels / COURT_LENGTH
        
        return {
            'pixels_per_meter_x': pixels_per_meter_x,
            'pixels_per_meter_y': pixels_per_meter_y,
            'court_width_pixels': court_width_pixels,
            'court_height_pixels': court_height_pixels,
            'court_width_meters': DOUBLE_LINE_WIDTH,
            'court_height_meters': COURT_LENGTH
        }
    else:
        return {
            'pixels_per_meter_x': 100,
            'pixels_per_meter_y': 100,
            'court_width_pixels': 1000,
            'court_height_pixels': 2000,
            'court_width_meters': DOUBLE_LINE_WIDTH,
            'court_height_meters': COURT_LENGTH
        }

# Bounding box utilities
def get_center_of_bbox(bbox):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)

def get_foot_position(bbox):
    """Get foot position (bottom center) of bounding box"""
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)

def get_height_of_bbox(bbox):
    """Get height of bounding box"""
    return bbox[3] - bbox[1]

def measure_xy_distance(p1, p2):
    """Measure X and Y distances separately"""
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

class PlayerTracker:
    """Mock PlayerTracker class"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Initialized PlayerTracker with model: {model_path}")
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Mock player detection"""
        print(f"Running player detection on {len(frames)} frames")
        
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Loading player detections from {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("Creating mock player detections")
            # Return mock detections
            detections = []
            for i in range(len(frames)):
                frame_detections = {
                    1: [[100 + i, 200 + i*2, 200 + i, 400 + i*2]],  # Player 1
                    2: [[400 + i, 150 + i, 500 + i, 350 + i]]       # Player 2
                }
                detections.append(frame_detections)
            return detections
    
    def choose_and_filter_players(self, court_keypoints, player_detections):
        """Filter and choose the two main players based on court position"""
        print("Filtering and choosing players based on court position")
        
        # Calculate court boundaries from keypoints
        points = court_keypoints.reshape(-1, 2)
        
        # Get court boundaries (with some margin)
        court_left = np.min(points[:, 0]) - 50
        court_right = np.max(points[:, 0]) + 50
        court_top = np.min(points[:, 1]) - 50
        court_bottom = np.max(points[:, 1]) + 50
        
        # Track player statistics
        player_stats = {}
        
        for frame_idx, frame_detections in enumerate(player_detections):
            if frame_detections and isinstance(frame_detections, dict):
                for player_id, bbox in frame_detections.items():
                    if isinstance(bbox, list) and len(bbox) >= 4:
                        x1, y1, x2, y2 = bbox[:4]
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        # Check if player is inside court boundaries
                        inside_court = (court_left <= center_x <= court_right and 
                                      court_top <= center_y <= court_bottom)
                        
                        if player_id not in player_stats:
                            player_stats[player_id] = {
                                'total_detections': 0,
                                'inside_court_detections': 0,
                                'positions': []
                            }
                        
                        player_stats[player_id]['total_detections'] += 1
                        player_stats[player_id]['positions'].append((center_x, center_y))
                        
                        if inside_court:
                            player_stats[player_id]['inside_court_detections'] += 1
        
        # Calculate inside court ratio for each player
        for player_id in player_stats:
            if player_stats[player_id]['total_detections'] > 0:
                ratio = player_stats[player_id]['inside_court_detections'] / player_stats[player_id]['total_detections']
                player_stats[player_id]['inside_court_ratio'] = ratio
            else:
                player_stats[player_id]['inside_court_ratio'] = 0
        
        # Select the two players with highest inside court ratio and minimum detections
        valid_players = {pid: stats for pid, stats in player_stats.items() 
                        if stats['total_detections'] >= 10 and stats['inside_court_ratio'] >= 0.3}
        
        if len(valid_players) >= 2:
            # Sort by inside court ratio and total detections
            sorted_players = sorted(valid_players.items(), 
                                  key=lambda x: (x[1]['inside_court_ratio'], x[1]['total_detections']), 
                                  reverse=True)
            
            main_player_ids = [sorted_players[0][0], sorted_players[1][0]]
            print(f"Selected main players: {main_player_ids}")
            
            # Filter detections to only include main players and rename them
            filtered_detections = []
            player_id_mapping = {main_player_ids[0]: 1, main_player_ids[1]: 2}
            
            for frame_detections in player_detections:
                filtered_frame = {}
                if frame_detections and isinstance(frame_detections, dict):
                    for player_id, bbox in frame_detections.items():
                        if player_id in main_player_ids:
                            new_id = player_id_mapping[player_id]
                            filtered_frame[new_id] = bbox
                filtered_detections.append(filtered_frame)
            
            print(f"Player filtering complete:")
            for orig_id, new_id in player_id_mapping.items():
                stats = player_stats[orig_id]
                print(f"  Player {orig_id} -> Player {new_id}: {stats['inside_court_detections']}/{stats['total_detections']} inside court ({stats['inside_court_ratio']:.1%})")
            
            return filtered_detections
        else:
            print("Warning: Could not find two valid players inside court, returning original detections")
            return player_detections

class BallTracker:
    """Mock BallTracker class"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Initialized BallTracker with model: {model_path}")
    
    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Mock ball detection"""
        print(f"Running ball detection on {len(frames)} frames")
        
        if read_from_stub and stub_path and os.path.exists(stub_path):
            print(f"Loading ball detections from {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)
        else:
            print("Creating mock ball detections")
            # Create realistic ball trajectory
            detections = []
            for i in range(len(frames)):
                if i % 20 < 15:  # Ball visible 75% of the time
                    # Simulate ball movement with some trajectory
                    x = 300 + 200 * np.sin(i * 0.1)
                    y = 200 + 100 * np.sin(i * 0.05) + i * 0.5
                    detection = {1: [x-10, y-10, x+10, y+10]}
                else:
                    detection = {}  # Ball not detected
                detections.append(detection)
            return detections
    
    def interpolate_ball_positions(self, ball_positions):
        """Interpolate missing ball positions"""
        print("Interpolating missing ball positions")
        ball_positions_list = [x.get(1, []) for x in ball_positions]
        df_ball = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball = df_ball.interpolate().bfill()
        return [{1: x} for x in df_ball.to_numpy().tolist()]
    
    def get_ball_shot_frames(self, ball_positions):
        """Detect ball shot frames using trajectory analysis"""
        print("Detecting ball shot frames")
        ball_positions_list = [x.get(1, []) for x in ball_positions]
        df_ball = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        
        if len(df_ball) == 0:
            return []
        
        df_ball = df_ball.interpolate().bfill()
        df_ball['ball_hit'] = 0
        df_ball['mid_y'] = (df_ball['y1'] + df_ball['y2']) / 2
        df_ball['mid_y_rolling_mean'] = df_ball['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        df_ball['delta_y'] = df_ball['mid_y_rolling_mean'].diff()
        
        minimum_change_frames_for_hit = 25
        
        for i in range(1, len(df_ball) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = (df_ball['delta_y'].iloc[i] > 0 and 
                                      df_ball['delta_y'].iloc[i+1] < 0)
            positive_position_change = (df_ball['delta_y'].iloc[i] < 0 and 
                                      df_ball['delta_y'].iloc[i+1] > 0)
            
            if negative_position_change or positive_position_change:
                change_count = 0
                for change_frame in range(i+1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    if change_frame >= len(df_ball):
                        break
                    negative_following = (df_ball['delta_y'].iloc[i] > 0 and 
                                        df_ball['delta_y'].iloc[change_frame] < 0)
                    positive_following = (df_ball['delta_y'].iloc[i] < 0 and 
                                        df_ball['delta_y'].iloc[change_frame] > 0)
                    
                    if ((negative_position_change and negative_following) or 
                        (positive_position_change and positive_following)):
                        change_count += 1
                
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball.iloc[i, df_ball.columns.get_loc('ball_hit')] = 1
        
        return df_ball[df_ball['ball_hit'] == 1].index.tolist()

class CourtLineDetector:
    """Enhanced CourtLineDetector class with tennis court measurements"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        print(f"Initialized CourtLineDetector with model: {model_path}")
        
        # Tennis court constants (in meters)
        self.SINGLE_LINE_WIDTH = 8.23
        self.DOUBLE_LINE_WIDTH = 10.97
        self.HALF_COURT_LINE_HEIGHT = 11.88
        self.SERVICE_LINE_WIDTH = 6.4
        self.DOUBLE_ALLY_DIFFERENCE = 1.37
        self.NO_MANS_LAND_HEIGHT = 5.48
        
        # Player heights for scale reference
        self.PLAYER_1_HEIGHT_METERS = 1.88
        self.PLAYER_2_HEIGHT_METERS = 1.91
    
    def predict(self, frame):
        """Enhanced court keypoint prediction with realistic court layout"""
        print("Predicting court keypoints based on frame analysis")
        
        # Analyze frame dimensions for realistic court placement
        height, width = frame.shape[:2]
        
        # Create more realistic court keypoints based on typical tennis court perspective
        # These represent the 14 standard tennis court keypoints
        keypoints = np.array([
            # Baseline corners (points 0-3)
            [width * 0.15, height * 0.85],  # 0: Left baseline corner
            [width * 0.85, height * 0.85],  # 1: Right baseline corner  
            [width * 0.15, height * 0.15],  # 2: Left baseline corner (far)
            [width * 0.85, height * 0.15],  # 3: Right baseline corner (far)
            
            # Singles sideline points (points 4-7)
            [width * 0.25, height * 0.85],  # 4: Left singles line (near)
            [width * 0.25, height * 0.15],  # 5: Left singles line (far)
            [width * 0.75, height * 0.85],  # 6: Right singles line (near)
            [width * 0.75, height * 0.15],  # 7: Right singles line (far)
            
            # Service line corners (points 8-11)
            [width * 0.25, height * 0.65],  # 8: Left service line (near)
            [width * 0.75, height * 0.65],  # 9: Right service line (near)
            [width * 0.25, height * 0.35],  # 10: Left service line (far)
            [width * 0.75, height * 0.35],  # 11: Right service line (far)
            
            # Center service line points (points 12-13)
            [width * 0.50, height * 0.65],  # 12: Center service line (near)
            [width * 0.50, height * 0.35],  # 13: Center service line (far)
        ])
        
        return keypoints.flatten()  # Flatten to match expected format
    
    def get_court_measurements(self):
        """Return tennis court measurements in meters"""
        return {
            'single_line_width': self.SINGLE_LINE_WIDTH,
            'double_line_width': self.DOUBLE_LINE_WIDTH,
            'half_court_height': self.HALF_COURT_LINE_HEIGHT,
            'service_line_width': self.SERVICE_LINE_WIDTH,
            'double_alley_difference': self.DOUBLE_ALLY_DIFFERENCE,
            'no_mans_land_height': self.NO_MANS_LAND_HEIGHT
        }
    
    def calculate_court_dimensions_pixels(self, keypoints, frame_shape):
        """Calculate court dimensions in pixels from keypoints"""
        height, width = frame_shape[:2]
        
        # Extract key measurements from keypoints
        # Keypoints are flattened: [x0, y0, x1, y1, ...]
        points = keypoints.reshape(-1, 2)
        
        # Calculate court width in pixels (baseline)
        court_width_pixels = abs(points[1][0] - points[0][0])  # Right baseline - Left baseline
        
        # Calculate court height in pixels
        court_height_pixels = abs(points[0][1] - points[2][1])  # Near baseline - Far baseline
        
        # Calculate service box dimensions
        service_width_pixels = abs(points[9][0] - points[8][0])  # Service line width
        service_height_pixels = abs(points[8][1] - points[10][1])  # Service line height
        
        return {
            'court_width_pixels': court_width_pixels,
            'court_height_pixels': court_height_pixels,
            'service_width_pixels': service_width_pixels,
            'service_height_pixels': service_height_pixels,
            'pixels_per_meter_x': court_width_pixels / self.DOUBLE_LINE_WIDTH,
            'pixels_per_meter_y': court_height_pixels / (self.HALF_COURT_LINE_HEIGHT * 2)
        }

class MiniCourt:

    
    def __init__(self, frame):
        self.frame = frame
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20
        
        # Tennis court measurements (meters)
        self.SINGLE_LINE_WIDTH = 8.23
        self.DOUBLE_LINE_WIDTH = 10.97
        self.HALF_COURT_LINE_HEIGHT = 11.88
        self.SERVICE_LINE_WIDTH = 6.4
        self.DOUBLE_ALLY_DIFFERENCE = 1.37
        self.NO_MANS_LAND_HEIGHT = 5.48
        
        print("Initialized enhanced MiniCourt with court measurements")
        
        # Set up mini court layout
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        
    def set_canvas_background_box_position(self, frame):
        """Set the position of the mini court canvas on the frame"""
        frame = frame.copy()
        
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height
        
    def set_mini_court_position(self):
        """Set the actual court position within the canvas"""
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x
        
    def convert_meters_to_pixels(self, meters):
        """Convert real-world meters to mini court pixels"""
        return int(meters * self.court_drawing_width / self.DOUBLE_LINE_WIDTH)
        
    def set_court_drawing_key_points(self):
        """Set the 14 keypoints for the mini court layout"""
        drawing_key_points = [0] * 28
        
        # Point 0: Left baseline near
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # Point 1: Right baseline near  
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # Point 2: Left baseline far
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(self.HALF_COURT_LINE_HEIGHT * 2)
        # Point 3: Right baseline far
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        
        # Points 4-7: Singles lines
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(self.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(self.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(self.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(self.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        
        # Points 8-11: Service lines
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(self.NO_MANS_LAND_HEIGHT)
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(self.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(self.NO_MANS_LAND_HEIGHT)
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(self.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        
        # Points 12-13: Center service lines
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]
        
        self.drawing_key_points = drawing_key_points
        
    def set_court_lines(self):
        """Define which keypoints connect to form court lines"""
        self.lines = [
            (0, 2), (4, 5), (6, 7), (1, 3),  # Baselines and sidelines
            (0, 1), (8, 9), (10, 11), (2, 3)  # Additional court lines
        ]
        
    def draw_court(self, frame):
        """Draw the mini court on the frame"""
        # Draw keypoints
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            
        # Draw court lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
            
        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)
        
        return frame
        
    def draw_background_rectangle(self, frame):
        """Draw semi-transparent background for mini court"""
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]
        return out
        
    def convert_position_to_mini_court(self, position, court_keypoints=None, reference_scale=1.0):
        """Convert real court position to mini court coordinates with court measurements"""
        if court_keypoints is not None:
            # Use actual court keypoints for accurate conversion
            points = court_keypoints.reshape(-1, 2)
            
            # Calculate scaling factors based on court dimensions
            court_width_pixels = abs(points[1][0] - points[0][0])
            scale_x = self.court_drawing_width / court_width_pixels
            
            # Apply scaling and offset to mini court position
            mini_x = self.court_start_x + (position[0] - points[0][0]) * scale_x
            mini_y = self.court_start_y + (position[1] - points[0][1]) * scale_x
            
            return [mini_x, mini_y]
        else:
            # Fallback to simple scaling
            scale_x, scale_y = 0.2, 0.2
            return [self.court_start_x + position[0] * scale_x, self.court_start_y + position[1] * scale_y]
            
    def get_court_measurements(self):
        """Get tennis court measurements in meters"""
        return {
            'court_length': self.HALF_COURT_LINE_HEIGHT * 2,
            'court_width_doubles': self.DOUBLE_LINE_WIDTH,
            'court_width_singles': self.SINGLE_LINE_WIDTH,
            'service_line_width': self.SERVICE_LINE_WIDTH,
            'service_box_length': self.NO_MANS_LAND_HEIGHT,
            'alley_width': self.DOUBLE_ALLY_DIFFERENCE
        }

# Additional imports compatibility - create module-like objects
class TrackerModule:
    """Mock trackers module for compatibility"""
    PlayerTracker = PlayerTracker
    BallTracker = BallTracker

class UtilsModule:
    """Mock utils module for compatibility"""
    read_video = read_video
    save_video = save_video
    measure_distance = measure_distance
    draw_player_stats = draw_player_stats
    convert_pixel_distance_to_meters = convert_pixel_distance_to_meters

class BboxUtilsModule:
    """Mock bbox_utils module for compatibility"""
    get_center_of_bbox = get_center_of_bbox
    measure_distance = measure_distance
    get_foot_position = get_foot_position
    get_height_of_bbox = get_height_of_bbox
    measure_xy_distance = measure_xy_distance

# Create module instances for import compatibility
trackers = TrackerModule()
utils = UtilsModule()
bbox_utils = BboxUtilsModule()

__all__ = [
    'read_video', 'save_video', 'measure_distance', 'draw_player_stats',
    'convert_pixel_distance_to_meters', 'get_center_of_bbox', 'get_foot_position',
    'get_height_of_bbox', 'measure_xy_distance', 'PlayerTracker', 'BallTracker',
    'CourtLineDetector', 'MiniCourt', 'trackers', 'utils', 'bbox_utils'
]