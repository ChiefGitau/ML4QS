# Additional export enhancements for tennis dataset

def create_time_series_export(df_processed, export_timestamp):
    """Create time-series optimized export for temporal analysis"""
    
    # Create time-indexed dataset
    if 'timestamp' in df_processed.columns:
        df_timeseries = df_processed.set_index('timestamp')
        
        # Select key temporal features
        temporal_cols = [
            'ball_center_x', 'ball_center_y', 'ball_speed_kmh',
            'player_1_center_x', 'player_1_center_y', 'player_1_speed_kmh',
            'player_2_center_x', 'player_2_center_y', 'player_2_speed_kmh',
            'players_distance_meters', 'total_activity_speed'
        ]
        
        available_temporal_cols = [col for col in temporal_cols if col in df_timeseries.columns]
        df_temporal_export = df_timeseries[available_temporal_cols].copy()
        
        # Export time-series dataset
        timeseries_file = f'tennis_timeseries_ml4qs_{export_timestamp}.csv'
        df_temporal_export.to_csv(timeseries_file)
        
        return timeseries_file
    return None

def create_video_segmented_exports(df_processed, export_timestamp):
    """Create separate files for each video"""
    
    video_cols = [col for col in df_processed.columns if 'video' in col.lower() and 'source' in col.lower()]
    if not video_cols:
        return []
    
    video_col = video_cols[0]
    exported_files = []
    
    for video_name in df_processed[video_col].unique():
        if pd.notna(video_name):
            video_data = df_processed[df_processed[video_col] == video_name]
            
            # Clean video name for filename
            clean_name = str(video_name).replace('.mp4', '').replace('input_video', 'video')
            
            video_file = f'tennis_{clean_name}_ml4qs_{export_timestamp}.csv'
            video_data.to_csv(video_file, index=False)
            exported_files.append(video_file)
    
    return exported_files

def create_ml_ready_splits(df_processed, export_timestamp, test_size=0.2):
    """Create train/test splits ready for ML"""
    from sklearn.model_selection import train_test_split
    
    # Remove non-numeric and identifier columns
    exclude_cols = ['timestamp', 'video_transition', 'activity_intensity'] + \
                   [col for col in df_processed.columns if 'video' in col.lower() and 'source' in col.lower()]
    
    ml_cols = [col for col in df_processed.columns if col not in exclude_cols]
    numeric_cols = df_processed[ml_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    df_ml = df_processed[numeric_cols].fillna(0)  # Fill NaN for ML
    
    # Create train/test split
    train_data, test_data = train_test_split(df_ml, test_size=test_size, random_state=42)
    
    # Export ML-ready datasets
    train_file = f'tennis_train_ml4qs_{export_timestamp}.csv'
    test_file = f'tennis_test_ml4qs_{export_timestamp}.csv'
    
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)
    
    return train_file, test_file

def create_feature_importance_analysis(df_processed, export_timestamp):
    """Create feature importance analysis for key variables"""
    
    # Calculate feature statistics
    feature_stats = []
    
    for col in df_processed.select_dtypes(include=[np.number]).columns:
        stats = {
            'feature': col,
            'missing_rate': df_processed[col].isnull().mean(),
            'variance': df_processed[col].var(),
            'range': df_processed[col].max() - df_processed[col].min(),
            'mean': df_processed[col].mean(),
            'std': df_processed[col].std()
        }
        feature_stats.append(stats)
    
    df_feature_stats = pd.DataFrame(feature_stats)
    
    # Export feature analysis
    feature_analysis_file = f'tennis_feature_analysis_{export_timestamp}.csv'
    df_feature_stats.to_csv(feature_analysis_file, index=False)
    
    return feature_analysis_file

# Example usage:
# timeseries_file = create_time_series_export(df_processed, export_timestamp)
# video_files = create_video_segmented_exports(df_processed, export_timestamp) 
# train_file, test_file = create_ml_ready_splits(df_processed, export_timestamp)
# feature_file = create_feature_importance_analysis(df_processed, export_timestamp)