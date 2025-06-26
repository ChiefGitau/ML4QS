"""Data preprocessing helper functions for tennis analysis"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.linalg import solve
import warnings

def linear_interpolation(series, max_gap=30):
    """Linear interpolation with gap size limit"""
    interpolated = series.interpolate(method='linear', limit=max_gap)
    return interpolated

def cubic_spline_interpolation(series, max_gap=20):
    """Cubic spline interpolation for smooth trajectories"""
    try:
        interpolated = series.interpolate(method='spline', order=3, limit=max_gap)
        return interpolated
    except:
        return linear_interpolation(series, max_gap)

def advanced_spline_interpolation(series, max_gap=15):
    """Advanced cubic spline with boundary conditions"""
    try:
        valid_mask = ~series.isna()
        if valid_mask.sum() < 4:
            return cubic_spline_interpolation(series, max_gap)
        
        valid_indices = series.index[valid_mask]
        valid_values = series[valid_mask].values
        
        cs = CubicSpline(valid_indices, valid_values, bc_type='natural')
        
        result = series.copy()
        missing_indices = series.index[~valid_mask]
        
        for idx in missing_indices:
            closest_before = valid_indices[valid_indices <= idx]
            closest_after = valid_indices[valid_indices >= idx]
            
            if len(closest_before) > 0 and len(closest_after) > 0:
                gap_size = closest_after.min() - closest_before.max()
                if gap_size <= max_gap:
                    result.loc[idx] = cs(idx)
        
        return result
        
    except Exception as e:
        return cubic_spline_interpolation(series, max_gap)

def kalman_filter_imputation(series, process_variance=1.0, measurement_variance=1.0):
    """Kalman filter for trajectory imputation with position and velocity state"""
    try:
        n = len(series)
        if n < 2:
            return series
            
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        Q = np.array([[process_variance, 0], [0, process_variance]])
        R = np.array([[measurement_variance]])
        
        valid_indices = ~series.isna()
        if not valid_indices.any():
            return series
            
        first_valid_idx = valid_indices.idxmax()
        x = np.array([series.iloc[first_valid_idx], 0])
        P = np.eye(2) * 1.0
        
        result = series.copy()
        
        for i in range(1, n):
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            
            if pd.notna(series.iloc[i]):
                y = series.iloc[i] - H @ x_pred
                S = H @ P_pred @ H.T + R
                K = P_pred @ H.T / S
                
                x = x_pred + K * y
                P = P_pred - K @ H @ P_pred
            else:
                x = x_pred
                P = P_pred
                result.iloc[i] = x[0]
        
        return result
        
    except Exception as e:
        return cubic_spline_interpolation(series, max_gap=15)

def rolling_mean_imputation(series, window=10):
    """Fill missing values with rolling mean"""
    rolling_mean = series.rolling(window=window, min_periods=1, center=True).mean()
    filled = series.fillna(rolling_mean)
    return filled

def forward_fill_with_limit(series, limit=5):
    """Forward fill with limit for detection-based data"""
    return series.fillna(method='ffill', limit=limit)

def video_aware_imputation(df, series, video_column, method='cubic_spline'):
    """Apply imputation within each video separately"""
    result = series.copy()
    
    videos = df[video_column].unique()
    
    for video in videos:
        video_mask = df[video_column] == video
        video_data = series[video_mask]
        
        if len(video_data.dropna()) < 2:
            continue
            
        if method == 'kalman':
            imputed_data = kalman_filter_imputation(video_data)
        elif method == 'cubic_spline':
            imputed_data = cubic_spline_interpolation(video_data, max_gap=15)
        elif method == 'advanced_spline':
            imputed_data = advanced_spline_interpolation(video_data, max_gap=12)
        elif method == 'linear':
            imputed_data = linear_interpolation(video_data, max_gap=20)
        elif method == 'rolling_mean':
            imputed_data = rolling_mean_imputation(video_data, window=5)
        else:
            imputed_data = forward_fill_with_limit(video_data, limit=5)
        
        result[video_mask] = imputed_data
    
    return result

def rolling_mean_filter(data, window=5):
    """Simple rolling mean filter"""
    return data.rolling(window=window, min_periods=1, center=True).mean()

def exponential_smoothing(data, alpha=0.3):
    """Exponential weighted moving average"""
    return data.ewm(alpha=alpha, adjust=False).mean()

def savgol_smoothing(data, window=11, polyorder=3):
    """Savitzky-Golay filter for smooth derivatives"""
    try:
        valid_mask = ~data.isna()
        if valid_mask.sum() < window:
            return data
        
        smoothed = data.copy()
        if valid_mask.sum() >= window:
            valid_data = data[valid_mask]
            if len(valid_data) >= window:
                smoothed_values = savgol_filter(valid_data, window, polyorder)
                smoothed.loc[valid_mask] = smoothed_values
        
        return smoothed
    except:
        return rolling_mean_filter(data, window=5)

def median_filter(data, window=5):
    """Rolling median filter for outlier removal"""
    return data.rolling(window=window, min_periods=1, center=True).median()

def classify_detailed_court_zone(x_rel, y_rel, obj_type='ball'):
    """Classify position into detailed court zones"""
    if pd.isna(x_rel) or pd.isna(y_rel):
        return 'unknown'
    
    if y_rel < 0.1:
        zone = "behind_far_baseline"
    elif y_rel < 0.35:
        zone = "far_court"
    elif y_rel < 0.45:
        zone = "far_service_area"
    elif y_rel < 0.55:
        zone = "net_area"
    elif y_rel < 0.65:
        zone = "near_service_area"
    elif y_rel < 0.9:
        zone = "near_court"
    else:
        zone = "behind_near_baseline"
    
    if x_rel < 0.33:
        side = "left"
    elif x_rel > 0.67:
        side = "right"
    else:
        side = "center"
    
    return f"{zone}_{side}"

def validate_tennis_ranges(df, tennis_ranges):
    """Validate data against tennis-specific ranges"""
    range_violations = {}
    
    for var, (min_val, max_val) in tennis_ranges.items():
        if var in df.columns:
            if df[var].dtype.kind in 'biufc':
                data = df[var].dropna()
                
                if len(data) > 0:
                    below_min = (data < min_val).sum()
                    above_max = (data > max_val).sum()
                    total_violations = below_min + above_max
                    violation_rate = (total_violations / len(data)) * 100
                    
                    range_violations[var] = {
                        'below_min': below_min,
                        'above_max': above_max,
                        'total_violations': total_violations,
                        'violation_rate': violation_rate,
                        'data_range': (data.min(), data.max())
                    }
    
    return range_violations

def validate_court_boundaries(df):
    """Validate court boundary positions"""
    court_boundary_issues = {}
    objects_to_check = ['ball', 'player_1', 'player_2']
    
    for obj in objects_to_check:
        x_rel_candidates = [col for col in df.columns if f'{obj}' in col and 'court' in col and 'relative' in col and 'x' in col]
        y_rel_candidates = [col for col in df.columns if f'{obj}' in col and 'court' in col and 'relative' in col and 'y' in col]
        
        if x_rel_candidates and y_rel_candidates:
            x_rel = x_rel_candidates[0]
            y_rel = y_rel_candidates[0]
            
            if x_rel in df.columns and y_rel in df.columns:
                if df[x_rel].dtype.kind in 'biufc' and df[y_rel].dtype.kind in 'biufc':
                    x_data = df[x_rel].dropna()
                    y_data = df[y_rel].dropna()
                    
                    if len(x_data) > 0 and len(y_data) > 0:
                        margin = 0.1
                        
                        x_outside = ((x_data < -margin) | (x_data > 1 + margin)).sum()
                        y_outside = ((y_data < -margin) | (y_data > 1 + margin)).sum()
                        
                        total_outside = x_outside + y_outside
                        outside_rate = (total_outside / (len(x_data) + len(y_data))) * 100
                        
                        court_boundary_issues[obj] = {
                            'x_outside': x_outside,
                            'y_outside': y_outside,
                            'total_outside': total_outside,
                            'outside_rate': outside_rate
                        }
    
    return court_boundary_issues

def validate_temporal_consistency(df):
    """Check for unrealistic frame-to-frame changes"""
    temporal_issues = {}
    
    movement_vars = {
        'ball_center_x': 100,
        'ball_center_y': 100,
        'player_1_center_x': 50,
        'player_1_center_y': 50,
        'player_2_center_x': 50,
        'player_2_center_y': 50
    }
    
    for var, max_change in movement_vars.items():
        if var in df.columns:
            if df[var].dtype.kind in 'biufc':
                changes = df[var].diff().abs()
                large_jumps = (changes > max_change).sum()
                jump_rate = (large_jumps / len(changes.dropna())) * 100 if len(changes.dropna()) > 0 else 0
                
                temporal_issues[var] = {
                    'large_jumps': large_jumps,
                    'jump_rate': jump_rate,
                    'max_change_threshold': max_change
                }
    
    return temporal_issues

def analyze_missing_patterns(df, key_vars):
    """Analyze missing data patterns"""
    missing_patterns = {}
    
    for var in key_vars:
        if var in df.columns:
            is_missing = df[var].isna()
            
            if is_missing.any():
                missing_groups = is_missing.ne(is_missing.shift()).cumsum()[is_missing]
                if len(missing_groups) > 0:
                    gap_lengths = missing_groups.value_counts()
                    avg_gap_length = gap_lengths.mean() if len(gap_lengths) > 0 else 0
                    max_gap_length = gap_lengths.max() if len(gap_lengths) > 0 else 0
                    
                    missing_patterns[var] = {
                        'total_missing': is_missing.sum(),
                        'missing_rate': is_missing.mean() * 100,
                        'num_gaps': len(gap_lengths),
                        'avg_gap_length': avg_gap_length,
                        'max_gap_length': max_gap_length
                    }
    
    return missing_patterns

def calculate_quality_scores(range_violations, court_boundary_issues, temporal_issues, missing_patterns):
    """Calculate overall data quality scores"""
    range_score = 100
    if range_violations:
        avg_violation_rate = np.mean([v['violation_rate'] for v in range_violations.values()])
        range_score = max(0, 100 - avg_violation_rate * 2)
    
    boundary_score = 100
    if court_boundary_issues:
        avg_outside_rate = np.mean([v['outside_rate'] for v in court_boundary_issues.values()])
        boundary_score = max(0, 100 - avg_outside_rate * 5)
    
    temporal_score = 100
    if temporal_issues:
        avg_jump_rate = np.mean([v['jump_rate'] for v in temporal_issues.values()])
        temporal_score = max(0, 100 - avg_jump_rate * 10)
    
    missing_score = 100
    if missing_patterns:
        avg_missing_rate = np.mean([v['missing_rate'] for v in missing_patterns.values()])
        missing_score = max(0, 100 - avg_missing_rate)
    
    overall_quality = (range_score + boundary_score + temporal_score + missing_score) / 4
    
    return {
        'range_validation': range_score,
        'boundary_validation': boundary_score,
        'temporal_consistency': temporal_score,
        'missing_data_quality': missing_score,
        'overall_quality': overall_quality
    }

def apply_filtering_methods(df, filter_variables):
    """Apply multiple filtering methods to specified variables"""
    filtering_results = {}
    
    for group_name, variables in filter_variables.items():
        available_vars = [var for var in variables if var in df.columns]
        
        if not available_vars:
            continue
            
        for var in available_vars:
            original_data = df[var].copy()
            
            if original_data.dropna().empty:
                continue
                
            filters_applied = {}
            
            filtered_rolling = rolling_mean_filter(original_data, window=5)
            df[f"{var}_rolling_mean"] = filtered_rolling
            filters_applied['rolling_mean'] = filtered_rolling
            
            filtered_exp = exponential_smoothing(original_data, alpha=0.3)
            df[f"{var}_exponential"] = filtered_exp
            filters_applied['exponential'] = filtered_exp
            
            if group_name == 'positions':
                filtered_savgol = savgol_smoothing(original_data, window=11, polyorder=3)
                df[f"{var}_savgol"] = filtered_savgol
                filters_applied['savgol'] = filtered_savgol
            
            if group_name == 'speeds':
                filtered_median = median_filter(original_data, window=5)
                df[f"{var}_median"] = filtered_median
                filters_applied['median'] = filtered_median
            
            original_std = original_data.std()
            
            filter_stats = {}
            for filter_name, filtered_data in filters_applied.items():
                filtered_std = filtered_data.std()
                noise_reduction = ((original_std - filtered_std) / original_std * 100) if original_std > 0 else 0
                filter_stats[filter_name] = {
                    'noise_reduction_pct': noise_reduction,
                    'original_std': original_std,
                    'filtered_std': filtered_std
                }
            
            filtering_results[var] = filter_stats
    
    return filtering_results

def apply_scaling_methods(df, scaling_groups):
    """Apply multiple scaling methods to specified variable groups"""
    scalers = {
        'minmax': MinMaxScaler(),
        'standard': StandardScaler(), 
        'robust': RobustScaler()
    }
    
    scaling_stats = {}
    
    for group_name, variables in scaling_groups.items():
        available_vars = [var for var in variables if var in df.columns]
        
        if not available_vars:
            continue
            
        group_data = df[available_vars].copy()
        
        for scaler_name, scaler in scalers.items():
            try:
                group_data_clean = group_data.fillna(0)
                scaled_values = scaler.fit_transform(group_data_clean)
                
                scaled_df = pd.DataFrame(scaled_values, 
                                       columns=[f"{var}_{scaler_name}" for var in available_vars],
                                       index=group_data.index)
                
                for i, var in enumerate(available_vars):
                    scaled_col_name = f"{var}_{scaler_name}"
                    df[scaled_col_name] = scaled_df.iloc[:, i]
                    
                    original_data = group_data[var].dropna()
                    scaled_data_clean = scaled_df.iloc[:, i].dropna()
                    
                    if len(original_data) > 0 and len(scaled_data_clean) > 0:
                        scaling_stats[scaled_col_name] = {
                            'original_mean': original_data.mean(),
                            'original_std': original_data.std(),
                            'scaled_mean': scaled_data_clean.mean(),
                            'scaled_std': scaled_data_clean.std(),
                            'original_range': original_data.max() - original_data.min(),
                            'scaled_range': scaled_data_clean.max() - scaled_data_clean.min()
                        }
                
            except Exception as e:
                continue
    
    return scaling_stats