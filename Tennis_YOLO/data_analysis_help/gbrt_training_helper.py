"""Helper functions for GBRT model training and tennis ball trajectory prediction"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy.signal import savgol_filter
import pickle
import json
from datetime import datetime

def load_and_prepare_data(df):
    """Prepare data according to GBRT requirements"""
    print("Preparing data for GBRT model...")
    
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
    
    if 'ball_x1' in df.columns and 'ball_y1' in df.columns and 'ball_x2' in df.columns and 'ball_y2' in df.columns:
        df['ball_center_x_calc'] = (df['ball_x1'] + df['ball_x2']) / 2
        df['ball_center_y_calc'] = (df['ball_y1'] + df['ball_y2']) / 2
        
        if 'ball_center_x' not in df.columns or df['ball_center_x'].isnull().all():
            df['ball_center_x'] = df['ball_center_x_calc']
        if 'ball_center_y' not in df.columns or df['ball_center_y'].isnull().all():
            df['ball_center_y'] = df['ball_center_y_calc']
    
    required_cols = ['ball_center_x', 'ball_center_y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("   Applying interpolation and smoothing...")
    ball_cols = ['ball_center_x', 'ball_center_y']
    
    df[ball_cols] = df[ball_cols].interpolate(method='linear', limit=5).ffill().bfill()
    
    if len(df) > 11:
        window_size = min(11, len(df)//2)
        if window_size % 2 == 0: 
            window_size -= 1
        
        if window_size >= 3:
            try:
                df['ball_center_x'] = savgol_filter(df['ball_center_x'], window_size, 2)
                df['ball_center_y'] = savgol_filter(df['ball_center_y'], window_size, 2)
                print(f"   Applied Savitzky-Golay smoothing (window={window_size})")
            except Exception as e:
                print(f"   Savitzky-Golay smoothing failed: {e}")
    
    if 'court_width_pixels' not in df.columns:
        df['court_width_pixels'] = 1344.0
    if 'court_height_pixels' not in df.columns:
        df['court_height_pixels'] = 756.0
    
    df['court_width_pixels'] = df['court_width_pixels'].fillna(1344.0)
    df['court_height_pixels'] = df['court_height_pixels'].fillna(756.0)
    
    if 'time_seconds' not in df.columns:
        if 'global_frame_number' in df.columns:
            df['time_seconds'] = df['global_frame_number'] / 30.0
        else:
            df['time_seconds'] = np.arange(len(df)) / 30.0
    
    if 'player_1_center_x' not in df.columns:
        df['player_1_center_x'] = 500.0
    if 'player_1_center_y' not in df.columns:
        df['player_1_center_y'] = 400.0
    
    print(f"   Data preparation completed")
    print(f"   Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    return df

def create_features(df, window_size=5):
    """Create physics-based features for ball trajectory prediction"""
    print("Creating physics-based features for GBRT...")
    
    df_features = df.copy()
    
    time_diff = df_features['time_seconds'].diff().replace(0, 0.0333)
    
    print("   Creating court-relative positions...")
    df_features['rel_x'] = df_features['ball_center_x'] / df_features['court_width_pixels']
    df_features['rel_y'] = df_features['ball_center_y'] / df_features['court_height_pixels']
    
    print("   Calculating velocity components...")
    df_features['vx'] = df_features['ball_center_x'].diff() / time_diff
    df_features['vy'] = df_features['ball_center_y'].diff() / time_diff
    
    print("   Computing speed and acceleration...")
    df_features['speed'] = np.sqrt(df_features['vx']**2 + df_features['vy']**2)
    df_features['acceleration'] = np.sqrt(df_features['vx'].diff()**2 + df_features['vy'].diff()**2) / time_diff
    
    print("   Analyzing direction changes...")
    df_features['direction_change'] = np.arctan2(df_features['vy'], df_features['vx']).diff()
    
    print("   Calculating player distances...")
    df_features['player_dist'] = np.sqrt(
        (df_features['player_1_center_x'] - df_features['ball_center_x'])**2 + 
        (df_features['player_1_center_y'] - df_features['ball_center_y'])**2
    )
    
    print("   Computing moving averages...")
    for col in ['rel_x', 'rel_y', 'vx', 'vy']:
        df_features[f'{col}_ma3'] = df_features[col].rolling(3, min_periods=1).mean()
        df_features[f'{col}_ma5'] = df_features[col].rolling(window_size, min_periods=1).mean()
    
    print("   Creating court zone features...")
    df_features['near_net'] = ((df_features['rel_x'] > 0.4) & (df_features['rel_x'] < 0.6)).astype(int)
    df_features['in_corner'] = ((df_features['rel_y'] < 0.2) | (df_features['rel_y'] > 0.8)).astype(int)
    
    selected_features = [
        'rel_x', 'rel_y',
        'vx', 'vy',
        'speed', 'acceleration',
        'direction_change',
        'player_dist',
        'vx_ma3', 'vy_ma3',
        'rel_x_ma5', 'rel_y_ma5',
        'near_net', 'in_corner',
        'court_width_pixels', 'court_height_pixels'
    ]
    
    target_features = ['ball_center_x', 'ball_center_y']
    all_features = selected_features + target_features
    
    available_features = [col for col in all_features if col in df_features.columns]
    missing_features = [col for col in all_features if col not in df_features.columns]
    
    if missing_features:
        print(f"   Missing features: {missing_features}")
    
    result_df = df_features[available_features].dropna()
    
    print(f"   Feature engineering completed")
    print(f"   Features created: {len(selected_features)}")
    print(f"   Final dataset shape: {result_df.shape[0]:,} rows × {result_df.shape[1]} columns")
    
    if len(result_df) > 0:
        print(f"\nFeature Statistics:")
        stats_features = ['rel_x', 'rel_y', 'speed', 'acceleration', 'player_dist']
        for feature in stats_features:
            if feature in result_df.columns:
                mean_val = result_df[feature].mean()
                std_val = result_df[feature].std()
                print(f"   {feature}: mean={mean_val:.3f}, std={std_val:.3f}")
    
    return result_df

def prepare_sequences(df, hist_frames=12, pred_frames=5):
    """Prepare input sequences for GBRT time series prediction"""
    print(f"Preparing sequences for time series prediction...")
    print(f"   History frames: {hist_frames}")
    print(f"   Prediction frames: {pred_frames}")
    
    feature_cols = [col for col in df.columns if col not in ['ball_center_x', 'ball_center_y']]
    target_cols = ['ball_center_x', 'ball_center_y']
    
    print(f"   Feature columns: {len(feature_cols)}")
    print(f"   Target columns: {len(target_cols)}")
    
    X, y = [], []
    
    total_sequences = len(df) - hist_frames - pred_frames + 1
    print(f"   Total possible sequences: {total_sequences}")
    
    valid_sequences = 0
    
    for i in range(hist_frames, len(df) - pred_frames + 1):
        x_seq = df.iloc[i-hist_frames:i][feature_cols].values.flatten()
        y_seq = df.iloc[i:i+pred_frames][target_cols].values.flatten()
        
        if not np.isnan(x_seq).any() and not np.isnan(y_seq).any():
            X.append(x_seq)
            y.append(y_seq)
            valid_sequences += 1
    
    X, y = np.array(X), np.array(y)
    
    print(f"   Sequence preparation completed")
    print(f"   Valid sequences: {valid_sequences} / {total_sequences}")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    
    if len(X) == 0:
        raise ValueError("No valid training samples found. Check data preprocessing.")
    
    return X, y, feature_cols, target_cols

def train_physics_gbrt(X, y):
    """Train GBRT model with Bayesian hyperparameter optimization"""
    print("Training GBRT model with Bayesian optimization...")
    
    tscv = TimeSeriesSplit(n_splits=3, gap=3)
    print(f"   Cross-validation: {tscv.n_splits} splits with gap=3")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('imputer', SimpleImputer(strategy='median')),
        ('multi_gbrt', MultiOutputRegressor(
            GradientBoostingRegressor(
                loss='huber',
                alpha=0.95,
                random_state=42,
                n_iter_no_change=10,
                validation_fraction=0.15
            )
        ))
    ])
    
    params = {
        'multi_gbrt__estimator__learning_rate': Real(0.05, 0.15),
        'multi_gbrt__estimator__n_estimators': Integer(150, 250),
        'multi_gbrt__estimator__max_depth': Integer(4, 6),
        'multi_gbrt__estimator__min_samples_leaf': Integer(10, 30),
        'multi_gbrt__estimator__subsample': Real(0.7, 0.85)
    }
    
    print(f"   Hyperparameter search space:")
    for param, space in params.items():
        print(f"     {param}: {space}")
    
    opt = BayesSearchCV(
        estimator=pipeline,
        search_spaces=params,
        n_iter=30,
        cv=tscv,
        scoring='neg_mean_absolute_error',
        n_jobs=1,
        random_state=42,
        verbose=2
    )
    
    print(f"   Optimization iterations: 30")
    print(f"   Training data: {X.shape}")
    print(f"   Starting Bayesian optimization...")
    
    if len(X) > 0:
        sample_weights = np.geomspace(1.0, 3.0, num=len(X))
        print(f"   Using sample weights (recent data weighted higher)")
        opt.fit(X, y, multi_gbrt__sample_weight=sample_weights)
    else:
        opt.fit(X, y)
    
    print(f"   Optimization completed!")
    print(f"   Best CV score: {-opt.best_score_:.4f}")
    
    return opt.best_estimator_

def evaluate_model(model, X, y, target_cols):
    """Evaluate model performance and return metrics"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print(f"Data splits:")
    print(f"   Training: {X_train.shape[0]:,} sequences")
    print(f"   Testing: {X_test.shape[0]:,} sequences")
    
    print(f"\nMaking predictions...")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nPerformance Metrics:")
    print(f"   Mean Absolute Error (MAE): {mae:.2f} pixels")
    print(f"   Root Mean Square Error (RMSE): {rmse:.2f} pixels")
    print(f"   R² Score: {r2:.4f}")
    
    y_test_reshaped = y_test.reshape(-1, 5, 2)
    y_pred_reshaped = y_pred.reshape(-1, 5, 2)
    
    print(f"\nPer-Coordinate Performance:")
    
    x_mae = mean_absolute_error(y_test_reshaped[:, :, 0].flatten(), y_pred_reshaped[:, :, 0].flatten())
    x_rmse = np.sqrt(mean_squared_error(y_test_reshaped[:, :, 0].flatten(), y_pred_reshaped[:, :, 0].flatten()))
    x_r2 = r2_score(y_test_reshaped[:, :, 0].flatten(), y_pred_reshaped[:, :, 0].flatten())
    
    print(f"   X-coordinate: MAE={x_mae:.2f}px, RMSE={x_rmse:.2f}px, R²={x_r2:.4f}")
    
    y_mae = mean_absolute_error(y_test_reshaped[:, :, 1].flatten(), y_pred_reshaped[:, :, 1].flatten())
    y_rmse = np.sqrt(mean_squared_error(y_test_reshaped[:, :, 1].flatten(), y_pred_reshaped[:, :, 1].flatten()))
    y_r2 = r2_score(y_test_reshaped[:, :, 1].flatten(), y_pred_reshaped[:, :, 1].flatten())
    
    print(f"   Y-coordinate: MAE={y_mae:.2f}px, RMSE={y_rmse:.2f}px, R²={y_r2:.4f}")
    
    print(f"\nPerformance by Prediction Horizon:")
    horizon_maes = []
    for frame in range(5):
        frame_mae = mean_absolute_error(y_test_reshaped[:, frame, :].flatten(), y_pred_reshaped[:, frame, :].flatten())
        horizon_maes.append(frame_mae)
        print(f"   Frame {frame+1}: MAE={frame_mae:.2f} pixels")
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'x_mae': x_mae,
        'x_rmse': x_rmse,
        'x_r2': x_r2,
        'y_mae': y_mae,
        'y_rmse': y_rmse,
        'y_r2': y_r2,
        'horizon_maes': horizon_maes,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_test_reshaped': y_test_reshaped,
        'y_pred_reshaped': y_pred_reshaped
    }
    
    return metrics

def create_performance_plots(metrics, figsize=(15, 12)):
    """Create performance visualization plots"""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Tennis Ball Trajectory Prediction - GBRT Model Performance', fontsize=16, fontweight='bold')
    
    y_test = metrics['y_test']
    y_pred = metrics['y_pred']
    y_test_reshaped = metrics['y_test_reshaped']
    y_pred_reshaped = metrics['y_pred_reshaped']
    mae = metrics['mae']
    r2 = metrics['r2']
    x_mae = metrics['x_mae']
    y_mae = metrics['y_mae']
    x_r2 = metrics['x_r2']
    y_r2 = metrics['y_r2']
    horizon_maes = metrics['horizon_maes']
    
    # 1. Prediction vs Actual scatter plot
    ax1 = axes[0, 0]
    ax1.scatter(y_test.flatten(), y_pred.flatten(), alpha=0.5, s=1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual Position (pixels)')
    ax1.set_ylabel('Predicted Position (pixels)')
    ax1.set_title(f'Prediction vs Actual\nR² = {r2:.4f}')
    ax1.grid(True, alpha=0.3)
    
    # 2. Residuals plot
    ax2 = axes[0, 1]
    residuals = y_test.flatten() - y_pred.flatten()
    ax2.scatter(y_pred.flatten(), residuals, alpha=0.5, s=1)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted Position (pixels)')
    ax2.set_ylabel('Residuals (pixels)')
    ax2.set_title(f'Residuals Plot\nMAE = {mae:.2f} pixels')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance by prediction horizon
    ax3 = axes[1, 0]
    horizons = list(range(1, 6))
    ax3.bar(horizons, horizon_maes, color='skyblue', alpha=0.7)
    ax3.set_xlabel('Prediction Horizon (frames)')
    ax3.set_ylabel('MAE (pixels)')
    ax3.set_title('Prediction Accuracy by Horizon')
    ax3.grid(True, alpha=0.3)
    
    # 4. X vs Y coordinate performance
    ax4 = axes[1, 1]
    coordinates = ['X-coordinate', 'Y-coordinate']
    coord_maes = [x_mae, y_mae]
    coord_r2s = [x_r2, y_r2]
    
    x_pos = np.arange(len(coordinates))
    bars = ax4.bar(x_pos, coord_maes, color=['lightcoral', 'lightgreen'], alpha=0.7)
    ax4.set_xlabel('Coordinate')
    ax4.set_ylabel('MAE (pixels)')
    ax4.set_title('Performance by Coordinate')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(coordinates)
    ax4.grid(True, alpha=0.3)
    
    for i, (bar, r2_val) in enumerate(zip(bars, coord_r2s)):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                 f'R²={r2_val:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_trajectory_plots(metrics, figsize=(15, 6)):
    """Create sample trajectory visualization plots"""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Sample Tennis Ball Trajectory Predictions', fontsize=16, fontweight='bold')
    
    y_test_reshaped = metrics['y_test_reshaped']
    y_pred_reshaped = metrics['y_pred_reshaped']
    
    sample_indices = np.random.choice(len(y_test_reshaped), size=3, replace=False)
    colors = ['blue', 'red', 'green']
    
    for i, (idx, color) in enumerate(zip(sample_indices, colors)):
        actual_traj = y_test_reshaped[idx]
        pred_traj = y_pred_reshaped[idx]
        
        axes[0].plot(range(5), actual_traj[:, 0], 'o-', color=color, label=f'Actual {i+1}', linewidth=2)
        axes[0].plot(range(5), pred_traj[:, 0], 's--', color=color, label=f'Predicted {i+1}', linewidth=2, alpha=0.7)
        
        axes[1].plot(range(5), actual_traj[:, 1], 'o-', color=color, label=f'Actual {i+1}', linewidth=2)
        axes[1].plot(range(5), pred_traj[:, 1], 's--', color=color, label=f'Predicted {i+1}', linewidth=2, alpha=0.7)
    
    axes[0].set_xlabel('Future Frame')
    axes[0].set_ylabel('X Position (pixels)')
    axes[0].set_title('X-Coordinate Trajectory')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Future Frame')
    axes[1].set_ylabel('Y Position (pixels)')
    axes[1].set_title('Y-Coordinate Trajectory')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def get_model_parameters(model):
    """Extract and display model parameters"""
    gbrt_params = model.named_steps['multi_gbrt'].estimator.get_params()
    
    print("OPTIMIZED MODEL PARAMETERS")
    print("=" * 50)
    
    print("Gradient Boosting Regressor Parameters:")
    key_params = ['learning_rate', 'n_estimators', 'max_depth', 'min_samples_leaf', 'subsample']
    for param in key_params:
        if param in gbrt_params:
            print(f"   {param}: {gbrt_params[param]}")
    
    print(f"\nPipeline components:")
    for step_name, step in model.named_steps.items():
        print(f"   {step_name}: {type(step).__name__}")
    
    return gbrt_params

def generate_model_summary(df_raw, df_features, X, feature_cols, video_cols, metrics):
    """Generate comprehensive model summary"""
    mae = metrics['mae']
    rmse = metrics['rmse']
    r2 = metrics['r2']
    x_mae = metrics['x_mae']
    y_mae = metrics['y_mae']
    
    if mae < 20:
        performance_level = "Excellent"
    elif mae < 40:
        performance_level = "Good"
    elif mae < 60:
        performance_level = "Fair"
    else:
        performance_level = "Needs Improvement"
    
    print("TENNIS BALL TRAJECTORY PREDICTION - MODEL SUMMARY")
    print("=" * 60)
    
    print(f"Dataset Information:")
    print(f"   Original dataset: {df_raw.shape[0]:,} frames from {df_raw[video_cols[0]].nunique()} videos")
    print(f"   Preprocessed dataset: {df_features.shape[0]:,} frames with {len(feature_cols)} features")
    print(f"   Training sequences: {len(X):,} sequences")
    
    print(f"\nModel Configuration:")
    print(f"   Algorithm: Gradient Boosting Regression Trees (GBRT)")
    print(f"   Multi-output: Yes (predicts x,y coordinates)")
    print(f"   Input sequence length: 12 frames")
    print(f"   Prediction horizon: 5 frames")
    print(f"   Features per frame: {len(feature_cols)}")
    print(f"   Hyperparameter optimization: Bayesian (30 iterations)")
    
    print(f"\nModel Performance:")
    print(f"   Overall MAE: {mae:.2f} pixels")
    print(f"   Overall RMSE: {rmse:.2f} pixels")
    print(f"   Overall R²: {r2:.4f}")
    print(f"   X-coordinate MAE: {x_mae:.2f} pixels")
    print(f"   Y-coordinate MAE: {y_mae:.2f} pixels")
    
    print(f"\nKey Features Used:")
    important_features = ['rel_x', 'rel_y', 'vx', 'vy', 'speed', 'acceleration', 'player_dist']
    for feature in important_features:
        if feature in feature_cols:
            print(f"   {feature}")
    
    print(f"\nModel Capabilities:")
    capabilities = [
        "Multi-frame trajectory prediction",
        "Physics-informed feature engineering",
        "Court-aware position normalization",
        "Player interaction modeling",
        "Temporal sequence learning",
        "Robust hyperparameter optimization"
    ]
    for capability in capabilities:
        print(f"   {capability}")
    
    print(f"\nPerformance Assessment: {performance_level}")
    print(f"   MAE of {mae:.2f} pixels represents {'excellent' if mae < 20 else 'good' if mae < 40 else 'fair'} prediction accuracy")
    print(f"   R² of {r2:.4f} indicates {'strong' if r2 > 0.8 else 'moderate' if r2 > 0.6 else 'weak'} predictive power")
    
    print(f"\nTennis ball trajectory prediction model ready for deployment!")
    
    return performance_level

def save_model_and_results(model, df_raw, df_features, X, feature_cols, video_cols, metrics):
    """Save model and results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_file = f'tennis_gbrt_model_{timestamp}.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved: {model_file}")
    
    mae = metrics['mae']
    rmse = metrics['rmse']
    r2 = metrics['r2']
    x_mae = metrics['x_mae']
    y_mae = metrics['y_mae']
    x_r2 = metrics['x_r2']
    y_r2 = metrics['y_r2']
    
    if mae < 20:
        performance_level = "Excellent"
    elif mae < 40:
        performance_level = "Good"
    elif mae < 60:
        performance_level = "Fair"
    else:
        performance_level = "Needs Improvement"
    
    results = {
        'timestamp': timestamp,
        'dataset_info': {
            'original_frames': int(df_raw.shape[0]),
            'processed_frames': int(df_features.shape[0]),
            'training_sequences': int(len(X)),
            'features_per_frame': len(feature_cols),
            'videos_count': int(df_raw[video_cols[0]].nunique()) if video_cols else 0
        },
        'model_config': {
            'algorithm': 'Gradient Boosting Regression Trees',
            'input_sequence_length': 12,
            'prediction_horizon': 5,
            'optimization_iterations': 30,
            'cross_validation_splits': 3
        },
        'performance': {
            'overall_mae': float(mae),
            'overall_rmse': float(rmse),
            'overall_r2': float(r2),
            'x_coordinate_mae': float(x_mae),
            'y_coordinate_mae': float(y_mae),
            'x_coordinate_r2': float(x_r2),
            'y_coordinate_r2': float(y_r2),
            'performance_level': performance_level
        },
        'features_used': feature_cols
    }
    
    results_file = f'tennis_gbrt_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_file}")
    
    print(f"\nGBRT Tennis Ball Trajectory Prediction Complete!")
    print(f"   Model file: {model_file}")
    print(f"   Results file: {results_file}")
    print(f"   Performance: {performance_level} (MAE: {mae:.2f} pixels)")
    
    return {
        'model_file': model_file,
        'results_file': results_file,
        'performance_level': performance_level
    }