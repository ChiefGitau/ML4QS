#!/usr/bin/env python3

import pandas as pd
import numpy as np
import glob
from datetime import datetime
import os

def load_base_dataset():
    """Load the comprehensive temporal dataset as base structure."""
    print("Loading base comprehensive temporal dataset...")
    
    # Find the most recent comprehensive dataset
    temporal_files = glob.glob('tennis_comprehensive_temporal_dataset*.csv')
    if not temporal_files:
        raise FileNotFoundError("No comprehensive temporal dataset found")
    
    base_file = sorted(temporal_files)[-1]  # Get most recent
    print(f"Using base file: {base_file}")
    
    df_base = pd.read_csv(base_file)
    print(f"Base dataset shape: {df_base.shape}")
    return df_base, base_file

def load_multivideo_datasets():
    """Load multi-video datasets for integration."""
    print("\nLoading multi-video datasets...")
    
    datasets = {}
    
    # Ball tracking
    ball_files = glob.glob('tennis_multivideo_dataset_*_ball_tracking.csv')
    if ball_files:
        ball_file = sorted(ball_files)[-1]
        datasets['ball_tracking'] = pd.read_csv(ball_file)
        print(f"Ball tracking: {ball_file} - {datasets['ball_tracking'].shape}")
    
    # Player tracking
    player_files = glob.glob('tennis_multivideo_dataset_*_player_tracking.csv')
    if player_files:
        player_file = sorted(player_files)[-1]
        datasets['player_tracking'] = pd.read_csv(player_file)
        print(f"Player tracking: {player_file} - {datasets['player_tracking'].shape}")
    
    # Video summary
    summary_files = glob.glob('tennis_multivideo_dataset_*_video_summary.csv')
    if summary_files:
        summary_file = sorted(summary_files)[-1]
        datasets['video_summary'] = pd.read_csv(summary_file)
        print(f"Video summary: {summary_file} - {datasets['video_summary'].shape}")
    
    return datasets

def load_enhanced_datasets():
    """Load enhanced datasets with court measurements."""
    print("\nLoading enhanced datasets...")
    
    enhanced = {}
    
    # Enhanced ball data
    if os.path.exists('tennis_ball_enhanced_with_court.csv'):
        enhanced['ball'] = pd.read_csv('tennis_ball_enhanced_with_court.csv')
        print(f"Enhanced ball: {enhanced['ball'].shape}")
    
    # Enhanced player data
    if os.path.exists('tennis_players_enhanced_with_court.csv'):
        enhanced['players'] = pd.read_csv('tennis_players_enhanced_with_court.csv')
        print(f"Enhanced players: {enhanced['players'].shape}")
    
    return enhanced

def create_composite_key(df, video_col='video_name', frame_col='global_frame_number'):
    """Create composite key for merging."""
    if video_col in df.columns and frame_col in df.columns:
        return df[video_col].astype(str) + '_' + df[frame_col].astype(str)
    elif frame_col in df.columns:
        return 'input_video.mp4_' + df[frame_col].astype(str)
    else:
        return df.index.astype(str)

def merge_datasets(df_base, multivideo_data, enhanced_data):
    """Strategically merge all datasets."""
    print("\nMerging datasets...")
    
    # Start with base dataset
    df_unified = df_base.copy()
    print(f"Starting with base dataset: {df_unified.shape}")
    
    # Add composite key to base
    df_unified['merge_key'] = create_composite_key(df_unified, 'video_name' if 'video_name' in df_unified.columns else None, 'global_frame_number')
    
    # Merge multi-video ball tracking
    if 'ball_tracking' in multivideo_data:
        df_ball = multivideo_data['ball_tracking'].copy()
        df_ball['merge_key'] = create_composite_key(df_ball, 'video_name', 'global_frame_number')
        
        # Select relevant columns to merge
        ball_cols = ['merge_key', 'x1', 'y1', 'x2', 'y2', 'center_x', 'center_y', 
                    'distance_pixels', 'distance_meters', 'speed_ms', 'speed_kmh']
        ball_cols = [col for col in ball_cols if col in df_ball.columns]
        
        # Rename to avoid conflicts
        rename_map = {col: f'multivideo_ball_{col}' for col in ball_cols if col != 'merge_key'}
        df_ball_select = df_ball[ball_cols].rename(columns=rename_map)
        
        df_unified = df_unified.merge(df_ball_select, on='merge_key', how='left')
        print(f"After ball tracking merge: {df_unified.shape}")
    
    # Merge enhanced ball data
    if 'ball' in enhanced_data:
        df_enhanced_ball = enhanced_data['ball'].copy()
        df_enhanced_ball['merge_key'] = create_composite_key(df_enhanced_ball, 'video_name', 'global_frame_number')
        
        # Select enhanced features
        enhanced_ball_cols = ['merge_key', 'court_relative_x', 'court_relative_y', 
                             'court_position_x_meters', 'court_position_y_meters',
                             'estimated_height_meters', 'distance_to_net_meters', 'court_region']
        enhanced_ball_cols = [col for col in enhanced_ball_cols if col in df_enhanced_ball.columns]
        
        # Rename to avoid conflicts
        rename_map = {col: f'enhanced_ball_{col}' for col in enhanced_ball_cols if col != 'merge_key'}
        df_enhanced_ball_select = df_enhanced_ball[enhanced_ball_cols].rename(columns=rename_map)
        
        df_unified = df_unified.merge(df_enhanced_ball_select, on='merge_key', how='left')
        print(f"After enhanced ball merge: {df_unified.shape}")
    
    # Merge enhanced player data
    if 'players' in enhanced_data:
        df_enhanced_players = enhanced_data['players'].copy()
        df_enhanced_players['merge_key'] = create_composite_key(df_enhanced_players, 'video_name', 'global_frame_number')
        
        # Aggregate player data by frame (since there can be multiple players per frame)
        player_agg = df_enhanced_players.groupby('merge_key').agg({
            'court_relative_x': ['mean', 'std'],
            'court_relative_y': ['mean', 'std'],
            'court_position_x_meters': ['mean', 'std'],
            'court_position_y_meters': ['mean', 'std'],
            'distance_to_center_relative': ['mean', 'std'],
            'distance_to_net_meters': ['mean', 'std']
        }).reset_index()
        
        # Flatten column names
        player_agg.columns = ['merge_key'] + [f'enhanced_players_{col[0]}_{col[1]}' for col in player_agg.columns[1:]]
        
        df_unified = df_unified.merge(player_agg, on='merge_key', how='left')
        print(f"After enhanced players merge: {df_unified.shape}")
    
    # Add video summary statistics
    if 'video_summary' in multivideo_data:
        df_summary = multivideo_data['video_summary'].copy()
        
        # Create video-level merge key
        df_unified['video_key'] = df_unified['video_name'] if 'video_name' in df_unified.columns else 'input_video.mp4'
        df_summary['video_key'] = df_summary['video_name'] if 'video_name' in df_summary.columns else 'input_video.mp4'
        
        # Select summary statistics
        summary_cols = ['video_key', 'total_frames', 'duration_seconds', 'fps',
                       'ball_detection_rate', 'player_detection_rate', 'avg_ball_speed_kmh']
        summary_cols = [col for col in summary_cols if col in df_summary.columns]
        
        df_summary_select = df_summary[summary_cols]
        df_unified = df_unified.merge(df_summary_select, on='video_key', how='left')
        print(f"After video summary merge: {df_unified.shape}")
        
        # Clean up temporary key
        df_unified.drop('video_key', axis=1, inplace=True)
    
    # Clean up merge key
    df_unified.drop('merge_key', axis=1, inplace=True)
    
    return df_unified

def organize_columns(df):
    """Organize columns in logical order."""
    print("\nOrganizing columns...")
    
    # Define column order priorities
    temporal_cols = [col for col in df.columns if any(x in col.lower() for x in ['timestamp', 'frame', 'time', 'video'])]
    ball_cols = [col for col in df.columns if 'ball' in col.lower()]
    player_cols = [col for col in df.columns if 'player' in col.lower()]
    court_cols = [col for col in df.columns if 'court' in col.lower()]
    enhanced_cols = [col for col in df.columns if 'enhanced' in col.lower()]
    multivideo_cols = [col for col in df.columns if 'multivideo' in col.lower()]
    summary_cols = [col for col in df.columns if any(x in col.lower() for x in ['total_frames', 'duration', 'fps', 'detection_rate'])]
    
    # Combine in logical order
    ordered_cols = []
    for col_group in [temporal_cols, ball_cols, player_cols, court_cols, enhanced_cols, multivideo_cols, summary_cols]:
        for col in col_group:
            if col not in ordered_cols:
                ordered_cols.append(col)
    
    # Add any remaining columns
    remaining_cols = [col for col in df.columns if col not in ordered_cols]
    ordered_cols.extend(remaining_cols)
    
    return df[ordered_cols]

def main():
    """Main execution function."""
    print("=== Tennis Dataset Combination Script ===")
    print(f"Started at: {datetime.now()}")
    
    try:
        # Load all datasets
        df_base, base_file = load_base_dataset()
        multivideo_data = load_multivideo_datasets()
        enhanced_data = load_enhanced_datasets()
        
        # Merge datasets
        df_unified = merge_datasets(df_base, multivideo_data, enhanced_data)
        
        # Organize columns
        df_unified = organize_columns(df_unified)
        
        # Generate output filename
        output_file = f"dataset/tennis_unified_comprehensive_dataset.csv"
        
        # Save unified dataset
        print(f"\nSaving unified dataset...")
        df_unified.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\n=== COMBINATION COMPLETE ===")
        print(f"Output file: {output_file}")
        print(f"Final dataset shape: {df_unified.shape}")
        print(f"Columns: {len(df_unified.columns)}")
        print(f"Rows: {len(df_unified)}")
        
        # Print data quality summary
        print(f"\n=== DATA QUALITY SUMMARY ===")
        print(f"Non-null values per column:")
        null_summary = df_unified.isnull().sum()
        for col in df_unified.columns[:10]:  # Show first 10 columns
            non_null = len(df_unified) - null_summary[col]
            percentage = (non_null / len(df_unified)) * 100
            print(f"  {col}: {non_null}/{len(df_unified)} ({percentage:.1f}%)")
        
        if len(df_unified.columns) > 10:
            print(f"  ... and {len(df_unified.columns) - 10} more columns")
        
        print(f"\nCombination completed successfully at: {datetime.now()}")
        
        return output_file
        
    except Exception as e:
        print(f"Error during combination: {str(e)}")
        raise

if __name__ == "__main__":
    output_file = main()