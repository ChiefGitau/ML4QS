"""Helper functions for tennis dataset analysis"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

def categorize_columns(df):
    """Categorize DataFrame columns by type"""
    column_categories = {
        'Temporal': [col for col in df.columns if any(x in col.lower() for x in ['frame', 'time', 'seconds'])],
        'Ball': [col for col in df.columns if col.startswith('ball_')],
        'Player 1': [col for col in df.columns if col.startswith('player_1_')],
        'Player 2': [col for col in df.columns if col.startswith('player_2_')],
        'Court': [col for col in df.columns if col.startswith('court_')],
        'Detection/State': [col for col in df.columns if any(x in col for x in ['detected', 'rate', 'transition'])],
    }
    return column_categories

def calculate_detection_rates(df):
    """Calculate detection rates for all objects"""
    detection_stats = {}
    
    if 'ball_detected' in df.columns:
        ball_detection_rate = df['ball_detected'].mean()
        detection_stats['ball'] = {
            'rate': ball_detection_rate,
            'count': df['ball_detected'].sum(),
            'total': len(df)
        }

    for player_id in [1, 2]:
        detected_col = f'player_{player_id}_detected'
        if detected_col in df.columns:
            player_detection_rate = df[detected_col].mean()
            detection_stats[f'player_{player_id}'] = {
                'rate': player_detection_rate,
                'count': df[detected_col].sum(),
                'total': len(df)
            }

    if 'both_players_detected' in df.columns:
        both_players_rate = df['both_players_detected'].mean()
        detection_stats['both_players'] = {
            'rate': both_players_rate,
            'count': df['both_players_detected'].sum()
        }

    if 'all_objects_detected' in df.columns:
        all_objects_rate = df['all_objects_detected'].mean()
        detection_stats['all_objects'] = {
            'rate': all_objects_rate,
            'count': df['all_objects_detected'].sum()
        }
    
    return detection_stats

def analyze_missing_data(df):
    """Analyze missing data patterns"""
    missing_data = df.isnull().sum()
    missing_percentages = (missing_data / len(df)) * 100
    
    key_tracking_cols = [col for col in df.columns if any(x in col for x in ['center_x', 'center_y', 'speed', 'distance'])]
    
    missing_info = {}
    for col in key_tracking_cols:
        if col in missing_percentages.index:
            missing_pct = missing_percentages[col]
            if missing_pct > 0:
                missing_info[col] = {
                    'percentage': missing_pct,
                    'count': missing_data[col],
                    'total': len(df)
                }
    
    return missing_info

def analyze_detection_continuity(df):
    """Analyze detection continuity patterns"""
    continuity_stats = {}
    
    if 'ball_detected' in df.columns:
        ball_streaks = df['ball_detected'].astype(int).diff().ne(0).cumsum()
        ball_streak_lengths = df.groupby(ball_streaks)['ball_detected'].sum()
        ball_detected_streaks = ball_streak_lengths[ball_streak_lengths > 0]
        
        if len(ball_detected_streaks) > 0:
            continuity_stats['ball'] = {
                'streak_count': len(ball_detected_streaks),
                'avg_length': ball_detected_streaks.mean(),
                'max_length': ball_detected_streaks.max(),
                'min_length': ball_detected_streaks.min()
            }
    
    return continuity_stats

def check_temporal_consistency(df):
    """Check frame rate consistency"""
    consistency_info = {}
    
    if 'time_seconds' in df.columns:
        time_diffs = df['time_seconds'].diff().dropna()
        expected_frame_time = 1/30  # 30 fps
        actual_avg_frame_time = time_diffs.mean()
        
        consistency_info = {
            'expected_interval': expected_frame_time,
            'actual_interval': actual_avg_frame_time,
            'is_consistent': abs(actual_avg_frame_time - expected_frame_time) < 0.001
        }
    
    return consistency_info

def calculate_movement_stats(df):
    """Calculate movement statistics for players and ball"""
    movement_stats = {}
    
    # Player stats
    for player_id in [1, 2]:
        speed_col = f'player_{player_id}_speed_kmh'
        distance_col = f'player_{player_id}_distance_meters'
        
        if speed_col in df.columns:
            player_speeds = df[speed_col].dropna()
            player_distances = df[distance_col].dropna() if distance_col in df.columns else pd.Series()
            
            if len(player_speeds) > 0:
                realistic_speeds = player_speeds[player_speeds <= 50]
                
                movement_stats[f'player_{player_id}'] = {
                    'count': len(realistic_speeds),
                    'mean_speed': realistic_speeds.mean(),
                    'median_speed': realistic_speeds.median(),
                    'max_speed': realistic_speeds.max(),
                    'std_speed': realistic_speeds.std(),
                    'total_distance': player_distances.sum() if len(player_distances) > 0 else 0
                }
    
    # Ball stats
    if 'ball_speed_kmh' in df.columns:
        ball_speeds = df['ball_speed_kmh'].dropna()
        ball_distances = df['ball_distance_meters'].dropna() if 'ball_distance_meters' in df.columns else pd.Series()
        
        if len(ball_speeds) > 0:
            realistic_ball_speeds = ball_speeds[ball_speeds <= 200]
            
            movement_stats['ball'] = {
                'count': len(realistic_ball_speeds),
                'mean_speed': realistic_ball_speeds.mean(),
                'median_speed': realistic_ball_speeds.median(),
                'max_speed': realistic_ball_speeds.max(),
                'std_speed': realistic_ball_speeds.std(),
                'total_distance': ball_distances.sum() if len(ball_distances) > 0 else 0
            }
    
    return movement_stats

def analyze_court_regions(df):
    """Analyze court region usage patterns"""
    region_stats = {}
    
    for player_id in [1, 2]:
        region_col = f'player_{player_id}_court_region'
        if region_col in df.columns:
            region_counts = df[region_col].value_counts()
            if len(region_counts) > 0:
                region_stats[f'player_{player_id}'] = {}
                for region, count in region_counts.items():
                    percentage = (count / len(df)) * 100
                    region_stats[f'player_{player_id}'][region] = {
                        'count': count,
                        'percentage': percentage
                    }
    
    return region_stats

def analyze_ball_trajectory(df):
    """Analyze ball trajectory characteristics"""
    trajectory_stats = {}
    
    if 'ball_center_x' in df.columns and 'ball_center_y' in df.columns:
        ball_data = df[['ball_center_x', 'ball_center_y']].dropna()
        
        if len(ball_data) > 1:
            x_range = ball_data['ball_center_x'].max() - ball_data['ball_center_x'].min()
            y_range = ball_data['ball_center_y'].max() - ball_data['ball_center_y'].min()
            
            ball_y_changes = ball_data['ball_center_y'].diff().dropna()
            upward_movement = (ball_y_changes < 0).sum()
            downward_movement = (ball_y_changes > 0).sum()
            
            trajectory_stats = {
                'x_range': x_range,
                'y_range': y_range,
                'center_x': ball_data['ball_center_x'].mean(),
                'center_y': ball_data['ball_center_y'].mean(),
                'upward_frames': upward_movement,
                'downward_frames': downward_movement
            }
    
    return trajectory_stats

def analyze_player_interactions(df):
    """Analyze player interaction patterns"""
    interaction_stats = {}
    
    if all(f'player_{i}_center_x' in df.columns for i in [1, 2]):
        player_data = df[['player_1_center_x', 'player_1_center_y', 
                         'player_2_center_x', 'player_2_center_y']].dropna()
        
        if len(player_data) > 0:
            player_distances = np.sqrt(
                (player_data['player_1_center_x'] - player_data['player_2_center_x'])**2 +
                (player_data['player_1_center_y'] - player_data['player_2_center_y'])**2
            )
            
            interaction_stats = {
                'frames_with_both': len(player_data),
                'avg_distance_pixels': player_distances.mean(),
                'min_distance_pixels': player_distances.min(),
                'max_distance_pixels': player_distances.max()
            }
            
            if 'court_pixels_per_meter_x' in df.columns:
                pixels_per_meter = df['court_pixels_per_meter_x'].iloc[0]
                if not pd.isna(pixels_per_meter) and pixels_per_meter > 0:
                    interaction_stats['avg_distance_meters'] = player_distances.mean() / pixels_per_meter
    
    return interaction_stats

def calculate_activity_score(df):
    """Calculate activity intensity score"""
    activity_score = pd.Series(0, index=df.index)
    
    if 'ball_detected' in df.columns:
        activity_score += df['ball_detected'].astype(int)
    
    for player_id in [1, 2]:
        detected_col = f'player_{player_id}_detected'
        if detected_col in df.columns:
            activity_score += df[detected_col].astype(int)
    
    activity_stats = {}
    if activity_score.sum() > 0:
        activity_stats = {
            'mean_score': activity_score.mean(),
            'high_activity_frames': (activity_score == 3).sum(),
            'high_activity_pct': (activity_score == 3).mean(),
            'medium_activity_frames': (activity_score == 2).sum(),
            'medium_activity_pct': (activity_score == 2).mean(),
            'low_activity_frames': (activity_score == 1).sum(),
            'low_activity_pct': (activity_score == 1).mean(),
            'no_activity_frames': (activity_score == 0).sum(),
            'no_activity_pct': (activity_score == 0).mean()
        }
    
    return activity_stats, activity_score

def analyze_speed_variance(df):
    """Analyze speed variance patterns"""
    variance_stats = {}
    
    for player_id in [1, 2]:
        speed_col = f'player_{player_id}_speed_kmh'
        if speed_col in df.columns:
            speeds = df[speed_col].dropna()
            realistic_speeds = speeds[speeds <= 50]
            
            if len(realistic_speeds) > 10:
                cv = realistic_speeds.std() / realistic_speeds.mean() if realistic_speeds.mean() > 0 else 0
                high_speed_threshold = realistic_speeds.quantile(0.75)
                high_speed_frames = (realistic_speeds > high_speed_threshold).sum()
                
                variance_stats[f'player_{player_id}'] = {
                    'coefficient_variation': cv,
                    'high_speed_threshold': high_speed_threshold,
                    'high_speed_frames': high_speed_frames,
                    'high_speed_percentage': high_speed_frames / len(realistic_speeds)
                }
    
    return variance_stats

def create_data_quality_plots(df, figsize=(16, 12)):
    """Create data quality visualization plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    sample_interval = max(1, len(df) // 1000)
    sample_df = df.iloc[::sample_interval].copy()
    
    # 1. Detection rates over time
    ax1.set_title('Object Detection Rates Over Time', fontsize=14, fontweight='bold')
    
    if len(sample_df) > 0:
        window_size = min(50, len(sample_df) // 10)
        
        if 'ball_detected' in sample_df.columns:
            ball_rolling = sample_df['ball_detected'].rolling(window=window_size, min_periods=1).mean()
            ax1.plot(sample_df['time_seconds'], ball_rolling, label='Ball Detection Rate', linewidth=2, color='orange')
        
        if 'player_1_detected' in sample_df.columns:
            player1_rolling = sample_df['player_1_detected'].rolling(window=window_size, min_periods=1).mean()
            ax1.plot(sample_df['time_seconds'], player1_rolling, label='Player 1 Detection Rate', linewidth=2, color='blue')
        
        if 'player_2_detected' in sample_df.columns:
            player2_rolling = sample_df['player_2_detected'].rolling(window=window_size, min_periods=1).mean()
            ax1.plot(sample_df['time_seconds'], player2_rolling, label='Player 2 Detection Rate', linewidth=2, color='red')
    
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Detection Rate')
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Missing data heatmap
    ax2.set_title('Missing Data Pattern', fontsize=14, fontweight='bold')
    
    key_cols = [col for col in df.columns if any(x in col for x in ['center_x', 'center_y', 'detected'])]
    if len(key_cols) > 0:
        heatmap_sample = df[key_cols].iloc[::max(1, len(df)//500)]
        missing_matrix = heatmap_sample.isnull().astype(int)
        
        im = ax2.imshow(missing_matrix.T, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
        ax2.set_yticks(range(len(key_cols)))
        ax2.set_yticklabels([col.replace('_', '\n') for col in key_cols], fontsize=8)
        ax2.set_xlabel('Time (sampled frames)')
        plt.colorbar(im, ax=ax2, label='Missing (1) / Present (0)')
    else:
        ax2.text(0.5, 0.5, 'No key columns found for missing data analysis', 
                 transform=ax2.transAxes, ha='center', va='center')
    
    # 3. Detection frequency distribution
    ax3.set_title('Detection Frequency Distribution', fontsize=14, fontweight='bold')
    
    detection_cols = [col for col in df.columns if col.endswith('_detected')]
    if detection_cols:
        detection_counts = []
        labels = []
        
        for col in detection_cols:
            count = df[col].sum()
            detection_counts.append(count)
            label = col.replace('_detected', '').replace('_', ' ').title()
            labels.append(label)
        
        bars = ax3.bar(labels, detection_counts, alpha=0.7, color=['orange', 'blue', 'red', 'green', 'purple'][:len(labels)])
        ax3.set_ylabel('Number of Detections')
        ax3.set_xlabel('Object Type')
        
        for bar, count in zip(bars, detection_counts):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(detection_counts)*0.01,
                    f'{count:,}', ha='center', va='bottom', fontsize=10)
        
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    else:
        ax3.text(0.5, 0.5, 'No detection columns found', 
                 transform=ax3.transAxes, ha='center', va='center')
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Data completeness over time
    ax4.set_title('Data Completeness Over Time', fontsize=14, fontweight='bold')
    
    if len(sample_df) > 0:
        window_size = min(50, len(sample_df) // 10)
        
        if 'all_objects_detected' in sample_df.columns:
            completeness_rolling = sample_df['all_objects_detected'].rolling(window=window_size, min_periods=1).mean()
            ax4.plot(sample_df['time_seconds'], completeness_rolling, label='All Objects', linewidth=3, color='green')
        
        if 'both_players_detected' in sample_df.columns:
            both_players_rolling = sample_df['both_players_detected'].rolling(window=window_size, min_periods=1).mean()
            ax4.plot(sample_df['time_seconds'], both_players_rolling, label='Both Players', linewidth=2, color='purple')
    
    ax4.set_xlabel('Time (seconds)')
    ax4.set_ylabel('Completeness Rate')
    ax4.set_ylim(0, 1.1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_movement_analysis_plots(df, figsize=(16, 12)):
    """Create movement analysis visualization plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    colors = ['blue', 'red', 'orange']
    
    # 1. Speed distribution comparison
    ax1.set_title('Speed Distribution Comparison', fontsize=14, fontweight='bold')
    
    labels = []
    speed_data = []
    
    for i, player_id in enumerate([1, 2]):
        speed_col = f'player_{player_id}_speed_kmh'
        if speed_col in df.columns:
            speeds = df[speed_col].dropna()
            realistic_speeds = speeds[speeds <= 50]
            if len(realistic_speeds) > 0:
                speed_data.append(realistic_speeds)
                labels.append(f'Player {player_id}')
    
    if 'ball_speed_kmh' in df.columns:
        ball_speeds = df['ball_speed_kmh'].dropna()
        realistic_ball_speeds = ball_speeds[ball_speeds <= 200]
        if len(realistic_ball_speeds) > 0:
            speed_data.append(realistic_ball_speeds)
            labels.append('Ball')
    
    if speed_data:
        ax1.hist(speed_data, bins=30, alpha=0.7, label=labels, color=colors[:len(speed_data)])
        ax1.set_xlabel('Speed (km/h)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No speed data available', transform=ax1.transAxes, ha='center', va='center')
    
    # 2. Speed over time
    ax2.set_title('Speed Over Time', fontsize=14, fontweight='bold')
    
    sample_df = df.iloc[::max(1, len(df)//2000)].copy()
    
    for player_id, color in zip([1, 2], ['blue', 'red']):
        speed_col = f'player_{player_id}_speed_kmh'
        if speed_col in sample_df.columns:
            speeds = sample_df[speed_col].dropna()
            realistic_speeds = speeds[speeds <= 50]
            
            if len(realistic_speeds) > 0:
                time_values = sample_df.loc[realistic_speeds.index, 'time_seconds']
                ax2.scatter(time_values, realistic_speeds, alpha=0.6, s=10, 
                           color=color, label=f'Player {player_id}')
    
    if 'ball_speed_kmh' in sample_df.columns:
        ball_speeds = sample_df['ball_speed_kmh'].dropna()
        realistic_ball_speeds = ball_speeds[ball_speeds <= 200]
        
        if len(realistic_ball_speeds) > 0:
            time_values = sample_df.loc[realistic_ball_speeds.index, 'time_seconds']
            ax2.scatter(time_values, realistic_ball_speeds, alpha=0.6, s=10, 
                       color='orange', label='Ball')
    
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Speed (km/h)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Court position heatmap
    ax3.set_title('Player Position Heatmap', fontsize=14, fontweight='bold')
    
    all_x_positions = []
    all_y_positions = []
    
    for player_id in [1, 2]:
        x_col = f'player_{player_id}_center_x'
        y_col = f'player_{player_id}_center_y'
        
        if x_col in df.columns and y_col in df.columns:
            player_data = df[[x_col, y_col]].dropna()
            if len(player_data) > 0:
                all_x_positions.extend(player_data[x_col].tolist())
                all_y_positions.extend(player_data[y_col].tolist())
    
    if all_x_positions and all_y_positions:
        h = ax3.hist2d(all_x_positions, all_y_positions, bins=30, cmap='YlOrRd', alpha=0.8)
        plt.colorbar(h[3], ax=ax3, label='Frequency')
        ax3.set_xlabel('X Position (pixels)')
        ax3.set_ylabel('Y Position (pixels)')
        ax3.invert_yaxis()
    else:
        ax3.text(0.5, 0.5, 'No position data available', transform=ax3.transAxes, ha='center', va='center')
    
    # 4. Movement distance distribution
    ax4.set_title('Frame-to-Frame Distance Distribution', fontsize=14, fontweight='bold')
    
    distance_data = []
    distance_labels = []
    
    for player_id in [1, 2]:
        distance_col = f'player_{player_id}_distance_meters'
        if distance_col in df.columns:
            distances = df[distance_col].dropna()
            realistic_distances = distances[distances <= 5]
            if len(realistic_distances) > 0:
                distance_data.append(realistic_distances)
                distance_labels.append(f'Player {player_id}')
    
    if 'ball_distance_meters' in df.columns:
        ball_distances = df['ball_distance_meters'].dropna()
        realistic_ball_distances = ball_distances[ball_distances <= 10]
        if len(realistic_ball_distances) > 0:
            distance_data.append(realistic_ball_distances)
            distance_labels.append('Ball')
    
    if distance_data:
        ax4.hist(distance_data, bins=30, alpha=0.7, label=distance_labels, 
                 color=colors[:len(distance_data)])
        ax4.set_xlabel('Distance per Frame (meters)')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No distance data available', transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    return fig

def create_temporal_pattern_plots(df, figsize=(16, 12)):
    """Create temporal pattern visualization plots"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
    
    sample_interval = max(1, len(df) // 2000)
    sample_df = df.iloc[::sample_interval].copy()
    
    # 1. Activity intensity over time
    ax1.set_title('Activity Intensity Over Time', fontsize=14, fontweight='bold')
    
    activity_score = pd.Series(0, index=sample_df.index)
    if 'ball_detected' in sample_df.columns:
        activity_score += sample_df['ball_detected'].astype(int)
    for player_id in [1, 2]:
        detected_col = f'player_{player_id}_detected'
        if detected_col in sample_df.columns:
            activity_score += sample_df[detected_col].astype(int)
    
    if activity_score.sum() > 0:
        window_size = min(50, len(sample_df) // 10)
        activity_smooth = activity_score.rolling(window=window_size, min_periods=1).mean()
        
        ax1.plot(sample_df['time_seconds'], activity_smooth, linewidth=2, color='purple')
        ax1.fill_between(sample_df['time_seconds'], activity_smooth, alpha=0.3, color='purple')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Activity Score (0-3)')
        ax1.set_ylim(0, 3.1)
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No activity data available', transform=ax1.transAxes, ha='center', va='center')
    
    # 2. Ball trajectory visualization
    ax2.set_title('Ball Trajectory Pattern', fontsize=14, fontweight='bold')
    
    if 'ball_center_x' in df.columns and 'ball_center_y' in df.columns:
        ball_data = df[['ball_center_x', 'ball_center_y']].dropna()
        
        if len(ball_data) > 0:
            ball_sample = ball_data.iloc[::max(1, len(ball_data)//1000)]
            
            scatter = ax2.scatter(ball_sample['ball_center_x'], ball_sample['ball_center_y'],
                                c=range(len(ball_sample)), cmap='viridis', alpha=0.6, s=20)
            
            ax2.plot(ball_sample['ball_center_x'], ball_sample['ball_center_y'], 
                    alpha=0.3, linewidth=1, color='gray')
            
            ax2.set_xlabel('X Position (pixels)')
            ax2.set_ylabel('Y Position (pixels)')
            ax2.invert_yaxis()
            
            cbar = plt.colorbar(scatter, ax=ax2)
            cbar.set_label('Time sequence')
    else:
        ax2.text(0.5, 0.5, 'No ball trajectory data available', transform=ax2.transAxes, ha='center', va='center')
    
    # 3. Player distance over time
    ax3.set_title('Distance Between Players Over Time', fontsize=14, fontweight='bold')
    
    if all(f'player_{i}_center_x' in sample_df.columns for i in [1, 2]):
        player_data = sample_df[['player_1_center_x', 'player_1_center_y', 
                               'player_2_center_x', 'player_2_center_y', 'time_seconds']].dropna()
        
        if len(player_data) > 0:
            distances = np.sqrt(
                (player_data['player_1_center_x'] - player_data['player_2_center_x'])**2 +
                (player_data['player_1_center_y'] - player_data['player_2_center_y'])**2
            )
            
            ax3.plot(player_data['time_seconds'], distances, linewidth=1, alpha=0.7, color='green')
            
            if len(distances) > 10:
                window_size = min(20, len(distances) // 5)
                distances_smooth = distances.rolling(window=window_size, min_periods=1).mean()
                ax3.plot(player_data['time_seconds'], distances_smooth, linewidth=3, color='darkgreen', label='Moving Average')
                ax3.legend()
            
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Distance (pixels)')
            ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No player distance data available', transform=ax3.transAxes, ha='center', va='center')
    
    # 4. Speed correlation analysis
    ax4.set_title('Player Speed Correlation', fontsize=14, fontweight='bold')
    
    if all(f'player_{i}_speed_kmh' in df.columns for i in [1, 2]):
        speed_data = df[['player_1_speed_kmh', 'player_2_speed_kmh']].dropna()
        
        speed_data = speed_data[(speed_data['player_1_speed_kmh'] <= 50) & 
                               (speed_data['player_2_speed_kmh'] <= 50)]
        
        if len(speed_data) > 0:
            speed_sample = speed_data.sample(n=min(1000, len(speed_data)))
            
            ax4.scatter(speed_sample['player_1_speed_kmh'], speed_sample['player_2_speed_kmh'], 
                       alpha=0.6, s=20, color='blue')
            
            correlation = speed_data['player_1_speed_kmh'].corr(speed_data['player_2_speed_kmh'])
            
            z = np.polyfit(speed_sample['player_1_speed_kmh'], speed_sample['player_2_speed_kmh'], 1)
            p = np.poly1d(z)
            ax4.plot(speed_sample['player_1_speed_kmh'], p(speed_sample['player_1_speed_kmh']), 
                    "r--", alpha=0.8, linewidth=2)
            
            ax4.set_xlabel('Player 1 Speed (km/h)')
            ax4.set_ylabel('Player 2 Speed (km/h)')
            ax4.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                    transform=ax4.transAxes, fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No speed correlation data available', transform=ax4.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    return fig

def export_analysis_results(summary_stats, insights, findings, dataset_file, analysis_timestamp=None):
    """Export analysis results to files"""
    if analysis_timestamp is None:
        analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    export_info = {}
    
    try:
        # Export summary statistics as JSON
        analysis_results = {
            'analysis_timestamp': analysis_timestamp,
            'dataset_file': dataset_file,
            'summary_statistics': summary_stats,
            'ml4qs_insights': insights,
            'key_findings': findings
        }
        
        results_filename = f'tennis_analysis_results_{analysis_timestamp}.json'
        with open(results_filename, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        export_info['json_file'] = results_filename
        
        # Create analysis summary text report
        report_filename = f'tennis_analysis_report_{analysis_timestamp}.txt'
        with open(report_filename, 'w') as f:
            f.write("TENNIS DATASET ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset File: {dataset_file}\n\n")
            
            for category, stats in summary_stats.items():
                f.write(f"{category}:\n")
                for metric, value in stats.items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")
            
            f.write("ML4QS INSIGHTS:\n")
            for insight in insights:
                f.write(f"  {insight}\n")
            
            f.write("\nKEY FINDINGS:\n")
            for i, finding in enumerate(findings, 1):
                f.write(f"  {i}. {finding}\n")
        
        export_info['report_file'] = report_filename
        export_info['success'] = True
        
    except Exception as e:
        export_info['success'] = False
        export_info['error'] = str(e)
    
    return export_info

def generate_summary_statistics(df, movement_stats, detection_stats, region_stats):
    """Generate comprehensive summary statistics"""
    summary_stats = {
        'Dataset Information': {
            'Total frames': f"{len(df):,}",
            'Duration': f"{df['time_seconds'].max():.1f} seconds ({df['time_seconds'].max()/60:.1f} minutes)",
            'Frame rate': "30 fps",
            'Columns': len(df.columns)
        }
    }
    
    # Detection performance
    detection_summary = {}
    if 'ball' in detection_stats:
        detection_summary['Ball detection rate'] = f"{detection_stats['ball']['rate']:.1%}"
        detection_summary['Ball frames'] = f"{detection_stats['ball']['count']:,}"
    
    for player_id in [1, 2]:
        player_key = f'player_{player_id}'
        if player_key in detection_stats:
            detection_summary[f'Player {player_id} detection rate'] = f"{detection_stats[player_key]['rate']:.1%}"
            detection_summary[f'Player {player_id} frames'] = f"{detection_stats[player_key]['count']:,}"
    
    if 'all_objects' in detection_stats:
        detection_summary['All objects detected'] = f"{detection_stats['all_objects']['rate']:.1%}"
    
    summary_stats['Detection Performance'] = detection_summary
    
    # Movement statistics
    movement_summary = {}
    for player_id in [1, 2]:
        player_key = f'player_{player_id}'
        if player_key in movement_stats:
            stats = movement_stats[player_key]
            movement_summary[f'Player {player_id} avg speed'] = f"{stats['mean_speed']:.1f} km/h"
            movement_summary[f'Player {player_id} max speed'] = f"{stats['max_speed']:.1f} km/h"
            movement_summary[f'Player {player_id} total distance'] = f"{stats['total_distance']:.1f} m"
    
    if 'ball' in movement_stats:
        ball_stats = movement_stats['ball']
        movement_summary['Ball avg speed'] = f"{ball_stats['mean_speed']:.1f} km/h"
        movement_summary['Ball max speed'] = f"{ball_stats['max_speed']:.1f} km/h"
        movement_summary['Ball total distance'] = f"{ball_stats['total_distance']:.1f} m"
    
    summary_stats['Movement Statistics'] = movement_summary
    
    # Court usage statistics
    if region_stats:
        court_summary = {}
        for player_id in [1, 2]:
            player_key = f'player_{player_id}'
            if player_key in region_stats:
                regions = region_stats[player_key]
                if regions:
                    most_common_region = max(regions.keys(), key=lambda x: regions[x]['percentage'])
                    percentage = regions[most_common_region]['percentage']
                    court_summary[f'Player {player_id} most common region'] = f"{most_common_region.replace('_', ' ').title()} ({percentage:.1f}%)"
        
        if court_summary:
            summary_stats['Court Usage'] = court_summary
    
    # Data quality metrics
    quality_summary = {
        'Missing data percentage': f"{(df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%"
    }
    
    if 'all_objects' in detection_stats:
        quality_summary['Complete frames (all objects)'] = f"{detection_stats['all_objects']['count']:,}"
    
    key_columns = [col for col in df.columns if any(x in col for x in ['center_x', 'center_y', 'detected'])]
    if key_columns:
        completeness = ((len(df) * len(key_columns)) - df[key_columns].isnull().sum().sum()) / (len(df) * len(key_columns))
        quality_summary['Key data completeness'] = f"{completeness:.1%}"
    
    summary_stats['Data Quality'] = quality_summary
    
    return summary_stats

def generate_insights_and_findings(detection_stats, movement_stats):
    """Generate ML4QS insights and key findings"""
    insights = [
        "Temporal Data Structure: Successfully created time-series dataset from video frames",
        "Multi-Object Tracking: Coordinated tracking of ball and multiple players",
        "Data Quality Assessment: Systematic analysis of missing data patterns",
        "Feature Engineering: Derived speed, distance, and region metrics from raw positions",
        "Real-World Calibration: Court measurements enable meaningful physical units",
        "Statistical Analysis: Movement patterns and performance metrics extracted"
    ]
    
    findings = []
    
    if 'ball' in detection_stats:
        ball_rate = detection_stats['ball']['rate']
        if ball_rate > 0.5:
            findings.append(f"High ball detection rate ({ball_rate:.1%}) indicates good tracking quality")
        else:
            findings.append(f"Moderate ball detection rate ({ball_rate:.1%}) suggests challenging tracking conditions")
    
    if 'all_objects' in detection_stats:
        complete_rate = detection_stats['all_objects']['rate']
        if complete_rate > 0.3:
            findings.append(f"Good multi-object tracking with {complete_rate:.1%} complete frames")
        else:
            findings.append(f"Partial object tracking with {complete_rate:.1%} complete frames")
    
    # Player comparison
    player_speeds = []
    for player_id in [1, 2]:
        player_key = f'player_{player_id}'
        if player_key in movement_stats:
            player_speeds.append((player_id, movement_stats[player_key]['mean_speed']))
    
    if len(player_speeds) == 2:
        if abs(player_speeds[0][1] - player_speeds[1][1]) > 2:
            faster_player = player_speeds[0][0] if player_speeds[0][1] > player_speeds[1][1] else player_speeds[1][0]
            findings.append(f"Player {faster_player} shows higher average movement speed")
        else:
            findings.append("Players show similar movement speeds")
    
    return insights, findings