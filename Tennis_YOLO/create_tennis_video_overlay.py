
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import os
from datetime import datetime
from tennis_utils import MiniCourt

class TennisVideoOverlayCreator:
    """
    Creates tennis videos with detection overlays and mini court visualization
    """
    
    def __init__(self, court_measurements=None):
        # Standard tennis court measurements (meters)
        self.court_measurements = court_measurements or {
            'single_line_width': 8.23,
            'double_line_width': 10.97,
            'half_court_height': 11.88,
            'service_line_width': 6.4,
            'double_alley_difference': 1.37,
            'no_mans_land_height': 5.48
        }
        
        # Colors for visualization
        self.colors = {
            'ball': (0, 255, 255),      # Yellow
            'player1': (0, 255, 0),     # Green
            'player2': (255, 0, 0),     # Blue
            'court_lines': (255, 255, 255),  # White
            'background': (34, 139, 34),     # Forest Green
            'text': (255, 255, 255),         # White
            'mini_court_bg': (0, 100, 0)     # Dark Green
        }
        
        # Mini court dimensions (pixels)
        self.mini_court_width = 300
        self.mini_court_height = int(self.mini_court_width * 0.5)  # Tennis court ratio
        
    def draw_court_lines(self, frame, mini_court_instance=None):
        """Draw tennis court lines on frame using MiniCourt functionality"""
        if mini_court_instance is not None:
            # Use the MiniCourt's court drawing functionality
            frame_with_court = mini_court_instance.draw_background_rectangle(frame.copy())
            frame_with_court = mini_court_instance.draw_court(frame_with_court)
            return frame_with_court
        else:
            # Fallback to basic court outline if no mini court available
            h, w = frame.shape[:2]
            margin = 50
            
            # Outer court boundary
            cv2.rectangle(frame, (margin, margin), (w-margin, h-margin), self.colors['court_lines'], 2)
            
            # Net line (center)
            net_y = h // 2
            cv2.line(frame, (margin, net_y), (w-margin, net_y), self.colors['court_lines'], 3)
            
            # Service lines
            service_offset = (h - 2*margin) // 4
            cv2.line(frame, (margin, margin + service_offset), (w-margin, margin + service_offset), self.colors['court_lines'], 2)
            cv2.line(frame, (margin, h - margin - service_offset), (w-margin, h - margin - service_offset), self.colors['court_lines'], 2)
            
            # Single court lines (if court is wide enough)
            if w > 400:
                single_margin = margin + (w - 2*margin) // 8
                cv2.line(frame, (single_margin, margin), (single_margin, h-margin), self.colors['court_lines'], 2)
                cv2.line(frame, (w-single_margin, margin), (w-single_margin, h-margin), self.colors['court_lines'], 2)
        
        return frame
    
    def draw_detection_box(self, frame, x1, y1, x2, y2, label, color, confidence=None):
        """Draw detection bounding box with label"""
        # Draw bounding box
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Prepare label text
        if confidence is not None:
            text = f"{label}: {confidence:.2f}"
        else:
            text = label
        
        # Draw label background
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (int(x1), int(y1-30)), (int(x1 + text_size[0] + 10), int(y1)), color, -1)
        
        # Draw label text
        cv2.putText(frame, text, (int(x1 + 5), int(y1 - 10)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def draw_center_point(self, frame, center_x, center_y, label, color, radius=8):
        """Draw center point with label"""
        # Draw center circle
        cv2.circle(frame, (int(center_x), int(center_y)), radius, color, -1)
        cv2.circle(frame, (int(center_x), int(center_y)), radius+2, (255, 255, 255), 2)
        
        # Draw label
        cv2.putText(frame, label, (int(center_x + 15), int(center_y - 15)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def create_mini_court(self, width=300, height=150):
        """Create mini court visualization"""
        mini_court = np.full((height, width, 3), self.colors['mini_court_bg'], dtype=np.uint8)
        
        # Court boundaries
        margin = 20
        court_w = width - 2*margin
        court_h = height - 2*margin
        
        # Outer boundary
        cv2.rectangle(mini_court, (margin, margin), (width-margin, height-margin), 
                     self.colors['court_lines'], 2)
        
        # Net line
        net_y = height // 2
        cv2.line(mini_court, (margin, net_y), (width-margin, net_y), 
                self.colors['court_lines'], 2)
        
        # Service lines
        service_offset = court_h // 4
        cv2.line(mini_court, (margin, margin + service_offset), 
                (width-margin, margin + service_offset), self.colors['court_lines'], 1)
        cv2.line(mini_court, (margin, height - margin - service_offset), 
                (width-margin, height - margin - service_offset), self.colors['court_lines'], 1)
        
        # Single court lines
        single_margin = margin + court_w // 6
        cv2.line(mini_court, (single_margin, margin), (single_margin, height-margin), 
                self.colors['court_lines'], 1)
        cv2.line(mini_court, (width-single_margin, margin), (width-single_margin, height-margin), 
                self.colors['court_lines'], 1)
        
        return mini_court
    
    def convert_to_mini_court_coords(self, x, y, frame_width, frame_height):
        """Convert full court coordinates to mini court coordinates"""
        # Simple proportional conversion
        margin = 20
        court_w = self.mini_court_width - 2*margin
        court_h = self.mini_court_height - 2*margin
        
        # Normalize to 0-1
        norm_x = x / frame_width
        norm_y = y / frame_height
        
        # Convert to mini court coordinates
        mini_x = margin + norm_x * court_w
        mini_y = margin + norm_y * court_h
        
        return int(mini_x), int(mini_y)
    
    def add_tracking_info_panel(self, combined_frame, ball_data, player1_data, player2_data, frame_number):
        """Add tracking information panel to the combined frame"""
        panel_height = 200
        panel_width = combined_frame.shape[1]
        
        # Create info panel
        info_panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(info_panel, f"Frame {frame_number} - Tracking Information", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
        
        # Ball information
        y_offset = 60
        if ball_data and not pd.isna(ball_data.get('center_x', np.nan)):
            cv2.putText(info_panel, "BALL TRACKING:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['ball'], 2)
            y_offset += 25
            cv2.putText(info_panel, f"Position: ({ball_data['center_x']:.1f}, {ball_data['center_y']:.1f})", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 20
            cv2.putText(info_panel, f"Confidence: {ball_data.get('confidence', 0.0):.3f}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 20
            if 'speed_kmh' in ball_data:
                cv2.putText(info_panel, f"Speed: {ball_data['speed_kmh']:.1f} km/h", 
                           (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        else:
            cv2.putText(info_panel, "BALL: Not detected", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Player 1 information
        y_offset = 60
        x_offset = 300
        if player1_data and not pd.isna(player1_data.get('center_x', np.nan)):
            cv2.putText(info_panel, "PLAYER 1:", (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['player1'], 2)
            y_offset += 25
            cv2.putText(info_panel, f"Position: ({player1_data['center_x']:.1f}, {player1_data['center_y']:.1f})", 
                       (x_offset + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 20
            cv2.putText(info_panel, f"Confidence: {player1_data.get('confidence', 0.0):.3f}", 
                       (x_offset + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        else:
            cv2.putText(info_panel, "PLAYER 1: Not detected", (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        # Player 2 information
        y_offset = 60
        x_offset = 600
        if player2_data and not pd.isna(player2_data.get('center_x', np.nan)):
            cv2.putText(info_panel, "PLAYER 2:", (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['player2'], 2)
            y_offset += 25
            cv2.putText(info_panel, f"Position: ({player2_data['center_x']:.1f}, {player2_data['center_y']:.1f})", 
                       (x_offset + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
            y_offset += 20
            cv2.putText(info_panel, f"Confidence: {player2_data.get('confidence', 0.0):.3f}", 
                       (x_offset + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 1)
        else:
            cv2.putText(info_panel, "PLAYER 2: Not detected", (x_offset, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        
        
        return info_panel
    
    def create_video_with_overlay(self, video_path, ball_data_df, player_data_df, 
                                 output_path, start_frame=0, num_frames=None):
        """
        Create video with tennis overlays and tracking information
        """
        
        print(f"Creating tennis video overlay...")
        print(f" Input: {os.path.basename(video_path)}")
        print(f" Output: {os.path.basename(output_path)}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Cideo properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Set processing range
        if num_frames is None:
            end_frame = total_frames
        else:
            end_frame = min(start_frame + num_frames, total_frames)
        
        frames_to_process = end_frame - start_frame
        print(f"Processing frames {start_frame} to {end_frame} ({frames_to_process} frames)")
        
        # Set start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read a sample frame to initialize MiniCourt
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        ret, sample_frame = cap.read()
        if not ret:
            raise ValueError("Cannot read sample frame for MiniCourt initialization")
        
        # Initialize MiniCourt with actual frame for proper court overlay
        mini_court_instance = MiniCourt(sample_frame)
        
        # Reset to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create mini court template
        mini_court_template = self.create_mini_court(self.mini_court_width, self.mini_court_height)
        
        # Calculate output dimensions
        # Layout: [Original Video | Mini Court]
        #         [   Tracking Information   ]
        main_width = frame_width + self.mini_court_width
        main_height = max(frame_height, self.mini_court_height)
        info_panel_height = 200
        output_width = main_width
        output_height = main_height + info_panel_height
        
        print(f"Output dimensions: {output_width}x{output_height}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened() and frame_count < frames_to_process:
            ret, frame = cap.read()
            if not ret:
                break
            
            current_frame_number = start_frame + frame_count
            
            # Create output frame
            output_frame = np.zeros((output_height, output_width, 3), dtype=np.uint8)
            
            # Copy original frame to left side
            output_frame[0:frame_height, 0:frame_width] = frame
            
            # Draw court lines on original frame using MiniCourt
            output_frame[0:frame_height, 0:frame_width] = self.draw_court_lines(
                output_frame[0:frame_height, 0:frame_width], mini_court_instance
            )
            
            # Get tracking data for current frame
            ball_data = self.get_frame_data(ball_data_df, current_frame_number)
            player1_data = self.get_frame_data(player_data_df, current_frame_number, player_id=1)
            player2_data = self.get_frame_data(player_data_df, current_frame_number, player_id=2)
            
            # Create mini court for this frame
            mini_court = mini_court_template.copy()
            
            # Draw ball detection on original frame and mini court
            if ball_data and not pd.isna(ball_data.get('center_x', np.nan)):
                # Original frame
                if all(k in ball_data for k in ['x1', 'y1', 'x2', 'y2']):
                    output_frame[0:frame_height, 0:frame_width] = self.draw_detection_box(
                        output_frame[0:frame_height, 0:frame_width],
                        ball_data['x1'], ball_data['y1'], ball_data['x2'], ball_data['y2'],
                        "Ball", self.colors['ball'], ball_data.get('confidence')
                    )
                
                output_frame[0:frame_height, 0:frame_width] = self.draw_center_point(
                    output_frame[0:frame_height, 0:frame_width],
                    ball_data['center_x'], ball_data['center_y'], "Ball", self.colors['ball']
                )
                
                # Mini court - use MiniCourt coordinate conversion
                try:
                    mini_pos = mini_court_instance.convert_position_to_mini_court(
                        (ball_data['center_x'], ball_data['center_y'])
                    )
                    mini_x, mini_y = int(mini_pos[0]), int(mini_pos[1])
                    
                    # Scale to our mini court template size
                    template_x = int(mini_x * self.mini_court_width / mini_court_instance.court_drawing_width)
                    template_y = int(mini_y * self.mini_court_height / (mini_court_instance.court_drawing_width * 0.5))
                    
                    cv2.circle(mini_court, (template_x, template_y), 4, self.colors['ball'], -1)
                    cv2.circle(mini_court, (template_x, template_y), 6, (255, 255, 255), 1)
                except:
                    # Fallback to simple conversion
                    mini_x, mini_y = self.convert_to_mini_court_coords(
                        ball_data['center_x'], ball_data['center_y'], frame_width, frame_height
                    )
                    cv2.circle(mini_court, (mini_x, mini_y), 4, self.colors['ball'], -1)
                    cv2.circle(mini_court, (mini_x, mini_y), 6, (255, 255, 255), 1)
            
            # Draw player detections
            for player_data, color, label in [
                (player1_data, self.colors['player1'], "Player 1"),
                (player2_data, self.colors['player2'], "Player 2")
            ]:
                if player_data and not pd.isna(player_data.get('center_x', np.nan)):
                    # Original frame
                    if all(k in player_data for k in ['x1', 'y1', 'x2', 'y2']):
                        output_frame[0:frame_height, 0:frame_width] = self.draw_detection_box(
                            output_frame[0:frame_height, 0:frame_width],
                            player_data['x1'], player_data['y1'], player_data['x2'], player_data['y2'],
                            label, color, player_data.get('confidence')
                        )
                    
                    output_frame[0:frame_height, 0:frame_width] = self.draw_center_point(
                        output_frame[0:frame_height, 0:frame_width],
                        player_data['center_x'], player_data['center_y'], label, color
                    )
                    
                    # Mini court - use MiniCourt coordinate conversion
                    try:
                        mini_pos = mini_court_instance.convert_position_to_mini_court(
                            (player_data['center_x'], player_data['center_y'])
                        )
                        mini_x, mini_y = int(mini_pos[0]), int(mini_pos[1])
                        
                        # Scale to our mini court template size
                        template_x = int(mini_x * self.mini_court_width / mini_court_instance.court_drawing_width)
                        template_y = int(mini_y * self.mini_court_height / (mini_court_instance.court_drawing_width * 0.5))
                        
                        cv2.circle(mini_court, (template_x, template_y), 6, color, -1)
                        cv2.circle(mini_court, (template_x, template_y), 8, (255, 255, 255), 1)
                    except:
                        # Fallback to simple conversion
                        mini_x, mini_y = self.convert_to_mini_court_coords(
                            player_data['center_x'], player_data['center_y'], frame_width, frame_height
                        )
                        cv2.circle(mini_court, (mini_x, mini_y), 6, color, -1)
                        cv2.circle(mini_court, (mini_x, mini_y), 8, (255, 255, 255), 1)
            
            # Add mini court to right side
            mini_start_x = frame_width
            mini_start_y = (main_height - self.mini_court_height) // 2
            output_frame[mini_start_y:mini_start_y + self.mini_court_height, 
                        mini_start_x:mini_start_x + self.mini_court_width] = mini_court
            
            # Add mini court title
            cv2.putText(output_frame, "Mini Court View", 
                       (mini_start_x + 10, mini_start_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['text'], 2)
            
            # Create and add tracking info panel
            info_panel = self.add_tracking_info_panel(
                output_frame, ball_data, player1_data, player2_data, current_frame_number
            )
            output_frame[main_height:main_height + info_panel_height, :] = info_panel
            
            # Write frame
            out.write(output_frame)
            
            frame_count += 1
            processed_frames += 1
            
            # Progress update
            if frame_count % (frames_to_process // 10) == 0:
                progress = (frame_count / frames_to_process) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{frames_to_process} frames)")
        
        # Cleanup
        cap.release()
        out.release()

        print(f"Processed {processed_frames} frames")
        print(f"Output saved to: {output_path}")
        
        return output_path
    
    def get_frame_data(self, df, frame_number, player_id=None):
        if df is None or len(df) == 0:
            return None
        
        # Filter by frame number
        frame_data = df[df['frame_number'] == frame_number]
        
        # Filter by player ID if specified
        if player_id is not None and 'player_id' in df.columns:
            frame_data = frame_data[frame_data['player_id'] == player_id]
        
        if len(frame_data) == 0:
            return None
        
        # Return first match as dictionary
        return frame_data.iloc[0].to_dict()

def create_sample_tennis_video_overlay(video_path, ball_csv_path=None, player_csv_path=None, 
                                     output_dir="output_videos", start_frame=0, num_frames=300):

    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tracking data
    ball_df = None
    player_df = None
    
    if ball_csv_path and os.path.exists(ball_csv_path):
        print(f" ball tracking data: {os.path.basename(ball_csv_path)}")
        ball_df = pd.read_csv(ball_csv_path)
        print(f"   Ball tracking records: {len(ball_df)}")
    
    if player_csv_path and os.path.exists(player_csv_path):
        print(f"player tracking data: {os.path.basename(player_csv_path)}")
        player_df = pd.read_csv(player_csv_path)
        print(f"   Player tracking records: {len(player_df)}")
    
    # Create overlay creator
    overlay_creator = TennisVideoOverlayCreator()
    
    # Generate output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_overlay_{timestamp}.mp4")
    
    # Create video with overlay
    result_path = overlay_creator.create_video_with_overlay(
        video_path=video_path,
        ball_data_df=ball_df,
        player_data_df=player_df,
        output_path=output_path,
        start_frame=start_frame,
        num_frames=num_frames
    )
    

    print(f"Output video: {result_path}")
    
    return result_path

if __name__ == "__main__":
    # Example usage
    sample_video = "input_videos/input_video.mp4"  # Replace with actual video path
    sample_ball_csv = "dataset/ball_tracking.csv"          # Replace with actual CSV path
    sample_player_csv = "dataset/player_tracking.csv"      # Replace with actual CSV path
    
    if os.path.exists(sample_video):
        create_sample_tennis_video_overlay(
            video_path=sample_video,
            ball_csv_path=sample_ball_csv if os.path.exists(sample_ball_csv) else None,
            player_csv_path=sample_player_csv if os.path.exists(sample_player_csv) else None,
            start_frame=0,
            num_frames=150  # Process 5 seconds at 30 FPS
        )
    else:
        print(f"Sample video not found: {sample_video}")
