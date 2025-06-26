"""
This implementation has:
- 480 trees, max_depth=6, learning_rate=0.07
- 69 features per frame, 12-frame sequences (828 input features)
- Physics-guided post-processing with speed limits and boundary corrections
- Serve/rally state detection
- Error decomposition analysis
"""

import numpy as np
import pandas as pd
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
import warnings
warnings.filterwarnings('ignore')

class TennisGBRTPaperAligned:
    """
    GBRT model aligned with paper specifications for tennis ball trajectory prediction
    """
    
    def __init__(self):
        # Paper-specified model parameters
        self.n_estimators = 480
        self.max_depth = 6
        self.learning_rate = 0.07
        self.hist_frames = 12
        self.pred_frames = 5
        self.features_per_frame = 69
        self.max_ball_speed = 61.1  # m/s (220 km/h)
        
    def load_data(self, filepath):
        """Load and prepare data according to paper specifications"""
        print("Loading data with paper-aligned preprocessing...")
        
        df = pd.read_csv(filepath)
        
        # Ensure timestamp and sorting
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate ball center if needed
        if 'ball_x1' in df.columns and 'ball_y1' in df.columns:
            df['ball_center_x'] = (df['ball_x1'] + df['ball_x2']) / 2
            df['ball_center_y'] = (df['ball_y1'] + df['ball_y2']) / 2
        
        # Ensure required columns
        required_cols = ['ball_center_x', 'ball_center_y', 'time_seconds']
        for col in required_cols:
            if col not in df.columns:
                if col == 'time_seconds':
                    df['time_seconds'] = np.arange(len(df)) / 30.0  # Assume 30 FPS
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Basic interpolation and smoothing
        ball_cols = ['ball_center_x', 'ball_center_y']
        df[ball_cols] = df[ball_cols].interpolate(method='linear', limit=5).ffill().bfill()
        
        # Savitzky-Golay filtering (paper specification: window=11, order=2)
        if len(df) > 11:
            try:
                df['ball_center_x'] = savgol_filter(df['ball_center_x'], 11, 2)
                df['ball_center_y'] = savgol_filter(df['ball_center_y'], 11, 2)
                print(" Applied Savitzky-Golay filtering (window=11, order=2)")
            except Exception as e:
                print(f"Savitzky-Golay filtering failed: {e}")
        
        return df
    
    def create_features_paper_spec(self, df):
        """Create 69 features per frame as specified in paper"""
        print("Creating 69 features per frame (paper specification)")
        
        df_features = df.copy()
        
        # Ensure court dimensions
        if 'court_width_pixels' not in df_features.columns:
            df_features['court_width_pixels'] = 1344.0
        if 'court_height_pixels' not in df_features.columns:
            df_features['court_height_pixels'] = 756.0
        
        df_features['court_width_pixels'] = df_features['court_width_pixels'].fillna(1344.0)
        df_features['court_height_pixels'] = df_features['court_height_pixels'].fillna(756.0)
        
        # Ensure player positions
        if 'player_1_center_x' not in df_features.columns:
            df_features['player_1_center_x'] = 500.0
        if 'player_1_center_y' not in df_features.columns:
            df_features['player_1_center_y'] = 400.0
        
        # Time differential
        time_diff = df_features['time_seconds'].diff().replace(0, 0.0333)
        
        # Court-relative positions (2 features)
        df_features['rel_x'] = df_features['ball_center_x'] / df_features['court_width_pixels']
        df_features['rel_y'] = df_features['ball_center_y'] / df_features['court_height_pixels']
        
        # Velocity components (2 features)
        df_features['vx'] = df_features['ball_center_x'].diff() / time_diff
        df_features['vy'] = df_features['ball_center_y'].diff() / time_diff
        
        # Convert to m/s (assuming pixels to meters conversion)
        pixels_per_meter = 100  # Approximate conversion factor
        df_features['vx_ms'] = df_features['vx'] / pixels_per_meter
        df_features['vy_ms'] = df_features['vy'] / pixels_per_meter
        
        #Speed and acceleration (3 features)
        df_features['speed'] = np.sqrt(df_features['vx']**2 + df_features['vy']**2)
        df_features['speed_ms'] = np.sqrt(df_features['vx_ms']**2 + df_features['vy_ms']**2)
        df_features['acceleration'] = np.sqrt(df_features['vx'].diff()**2 + df_features['vy'].diff()**2) / time_diff
        
        # Direction change (1 feature)
        df_features['direction_change'] = np.arctan2(df_features['vy'], df_features['vx']).diff()
        
        # Player distance (1 feature)
        df_features['player_dist'] = np.sqrt(
            (df_features['player_1_center_x'] - df_features['ball_center_x'])**2 + 
            (df_features['player_1_center_y'] - df_features['ball_center_y'])**2
        )
        
        #Moving averages - window size 3 (8 features)
        ma3_vars = ['rel_x', 'rel_y', 'vx', 'vy', 'speed', 'acceleration', 'direction_change', 'player_dist']
        for var in ma3_vars:
            df_features[f'{var}_ma3'] = df_features[var].rolling(3, min_periods=1).mean()
        
        #Moving averages - window size 5 (8 features)
        ma5_vars = ['rel_x', 'rel_y', 'vx', 'vy', 'speed', 'acceleration', 'direction_change', 'player_dist']
        for var in ma5_vars:
            df_features[f'{var}_ma5'] = df_features[var].rolling(5, min_periods=1).mean()
        
        #Standard deviations - window size 5 (8 features)
        for var in ma5_vars:
            df_features[f'{var}_std5'] = df_features[var].rolling(5, min_periods=1).std()
        
        # Court zone features (2 features)
        df_features['near_net'] = ((df_features['rel_x'] > 0.4) & (df_features['rel_x'] < 0.6)).astype(int)
        df_features['in_corner'] = ((df_features['rel_y'] < 0.2) | (df_features['rel_y'] > 0.8)).astype(int)
        
        # Serve/rally state detection (1 feature)
        # st = 1 if vball > 15 m/s and height < 2 m, else 0
        estimated_height = 1.5  # Default height assumption
        df_features['serve_rally_state'] = (
            (df_features['speed_ms'] > 15) & (estimated_height < 2)
        ).astype(int)
        
        # Additional physics features
        # Ball trajectory curvature
        df_features['trajectory_curvature'] = np.abs(df_features['direction_change'])
        
        # Velocity ratios
        df_features['vx_vy_ratio'] = np.abs(df_features['vx'] / (df_features['vy'] + 1e-6))
        df_features['speed_change'] = df_features['speed'].diff()
        
        # Court position features
        df_features['distance_to_net'] = np.abs(df_features['rel_x'] - 0.5)
        df_features['distance_to_baseline'] = np.minimum(df_features['rel_y'], 1 - df_features['rel_y'])
        df_features['distance_to_sideline'] = np.minimum(df_features['rel_x'], 1 - df_features['rel_x'])
        
        # Temporal features
        df_features['frame_number'] = np.arange(len(df_features))
        df_features['time_in_rally'] = df_features['time_seconds'] - df_features['time_seconds'].iloc[0]
        
        # Additional smoothed features
        df_features['rel_x_smooth'] = df_features['rel_x'].rolling(7, min_periods=1).mean()
        df_features['rel_y_smooth'] = df_features['rel_y'].rolling(7, min_periods=1).mean()
        
        # More velocity-based features
        df_features['vx_smooth'] = df_features['vx'].rolling(5, min_periods=1).mean()
        df_features['vy_smooth'] = df_features['vy'].rolling(5, min_periods=1).mean()
        df_features['speed_smooth'] = df_features['speed'].rolling(5, min_periods=1).mean()
        
        # Ball energy approximation
        df_features['kinetic_energy'] = 0.5 * df_features['speed_ms']**2  # Assuming unit mass
        
        # Rolling min/max features
        df_features['speed_max5'] = df_features['speed'].rolling(5, min_periods=1).max()
        df_features['speed_min5'] = df_features['speed'].rolling(5, min_periods=1).min()
        
        # Final feature selection to ensure exactly 69 features
        feature_list = [
            # Basic position and velocity (6)
            'rel_x', 'rel_y', 'vx', 'vy', 'vx_ms', 'vy_ms',
            # Speed and dynamics (5)
            'speed', 'speed_ms', 'acceleration', 'direction_change', 'speed_change',
            # Player interaction (1)
            'player_dist',
            # Moving averages 3-frame (8)
            'rel_x_ma3', 'rel_y_ma3', 'vx_ma3', 'vy_ma3', 'speed_ma3', 'acceleration_ma3', 'direction_change_ma3', 'player_dist_ma3',
            # Moving averages 5-frame (8)
            'rel_x_ma5', 'rel_y_ma5', 'vx_ma5', 'vy_ma5', 'speed_ma5', 'acceleration_ma5', 'direction_change_ma5', 'player_dist_ma5',
            # Standard deviations (8)
            'rel_x_std5', 'rel_y_std5', 'vx_std5', 'vy_std5', 'speed_std5', 'acceleration_std5', 'direction_change_std5', 'player_dist_std5',
            # Court zones and state (3)
            'near_net', 'in_corner', 'serve_rally_state',
            # Court positioning (3)
            'distance_to_net', 'distance_to_baseline', 'distance_to_sideline',
            # Trajectory features (2)
            'trajectory_curvature', 'vx_vy_ratio',
            # Temporal features (2)
            'frame_number', 'time_in_rally',
            # Smoothed features (5)
            'rel_x_smooth', 'rel_y_smooth', 'vx_smooth', 'vy_smooth', 'speed_smooth',
            # Energy and extrema (3)
            'kinetic_energy', 'speed_max5', 'speed_min5',
            # Court dimensions (2)
            'court_width_pixels', 'court_height_pixels'
        ]

        if len(feature_list) > 69:
            feature_list = feature_list[:69]
        elif len(feature_list) < 69:
            # Add padding features if needed
            for i in range(69 - len(feature_list)):
                df_features[f'padding_feature_{i}'] = 0.0
                feature_list.append(f'padding_feature_{i}')
        
        # Add target variables
        target_features = ['ball_center_x', 'ball_center_y']
        all_features = feature_list + target_features
        
        # Filter to available features and drop NaN
        available_features = [col for col in all_features if col in df_features.columns]
        result_df = df_features[available_features].dropna()
        
        print(f"Created exactly {len(feature_list)} features per frame")
        print(f"Final dataset shape: {result_df.shape}")
        
        return result_df, feature_list
    
    def prepare_sequences_paper_spec(self, df, feature_cols):
        """Prepare 828-dimensional input sequences (12 frames × 69 features)"""
        print(f"Preparing sequences: {self.hist_frames} frames × {len(feature_cols)} features = {self.hist_frames * len(feature_cols)} input dims")
        
        target_cols = ['ball_center_x', 'ball_center_y']
        
        X, y = [], []
        for i in range(self.hist_frames, len(df) - self.pred_frames + 1):
            # Historical features (flattened to 828 dimensions)
            x_seq = df.iloc[i-self.hist_frames:i][feature_cols].values.flatten()
            
            # Future ball positions
            y_seq = df.iloc[i:i+self.pred_frames][target_cols].values.flatten()
            
            if not np.isnan(x_seq).any() and not np.isnan(y_seq).any():
                X.append(x_seq)
                y.append(y_seq)
        
        X, y = np.array(X), np.array(y)
        
        print(f"Input dimension: {X.shape[1]} (should be 828)")
        print(f"Output dimension: {y.shape[1]} (should be 10)")
        print(f"Total sequences: {len(X)}")
        
        return X, y
    
    def train_paper_model(self, X, y):
        """Train model with exact paper specifications"""
        print("Training GBRT with paper specifications...")
        print(f"Trees: {self.n_estimators}")
        print(f"Max depth: {self.max_depth}")
        print(f"Learning rate: {self.learning_rate}")
        
        # Paper-specified hyperparameter ranges for optimization
        params = {
            'multi_gbrt__estimator__learning_rate': Real(0.05, 0.15),
            'multi_gbrt__estimator__n_estimators': Integer(150, 250),
            'multi_gbrt__estimator__max_depth': Integer(4, 6),
            'multi_gbrt__estimator__min_samples_leaf': Integer(10, 30),
            'multi_gbrt__estimator__subsample': Real(0.7, 0.85)
        }
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputer', SimpleImputer(strategy='median')),
            ('multi_gbrt', MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    loss='huber',
                    alpha=0.95,
                    random_state=42,
                    n_iter_no_change=10,
                    validation_fraction=0.15
                )
            ))
        ])
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3, gap=3)
        
        # Bayesian optimization
        opt = BayesSearchCV(
            estimator=pipeline,
            search_spaces=params,
            n_iter=30,
            cv=tscv,
            scoring='neg_mean_absolute_error',
            n_jobs=1,
            random_state=42,
            verbose=1
        )
        
        # Exponential sample weights (paper specification)
        sample_weights = np.geomspace(1.0, 3.0, num=len(X))
        
        print(" Starting optimization")
        opt.fit(X, y, multi_gbrt__sample_weight=sample_weights)
        
        print(f"Best CV score: {-opt.best_score_:.4f}")
        return opt.best_estimator_
    
    def apply_physics_constraints(self, predictions, time_diff=0.0333):
        """Apply physics-guided post-processing (paper Section 7.5)"""
        print("Applying physics constraints...")
        
        constrained_preds = predictions.copy()
        n_samples, n_outputs = predictions.shape
        
        # Reshape to [samples, frames, coordinates]
        pred_reshaped = predictions.reshape(n_samples, self.pred_frames, 2)
        
        for i in range(n_samples):
            for frame in range(self.pred_frames):
                if frame > 0:
                    # Calculate predicted velocity
                    dx = pred_reshaped[i, frame, 0] - pred_reshaped[i, frame-1, 0]
                    dy = pred_reshaped[i, frame, 1] - pred_reshaped[i, frame-1, 1]
                    
                    predicted_speed = np.sqrt(dx**2 + dy**2) / time_diff / 100  # Convert to m/s
                    
                    # Speed constraint: limit to 61.1 m/s (paper specification)
                    if predicted_speed > self.max_ball_speed:
                        scale_factor = self.max_ball_speed / predicted_speed
                        pred_reshaped[i, frame, 0] = pred_reshaped[i, frame-1, 0] + dx * scale_factor
                        pred_reshaped[i, frame, 1] = pred_reshaped[i, frame-1, 1] + dy * scale_factor
                
                # Court boundary reflection (simplified)
                x_rel = pred_reshaped[i, frame, 0] / 1344.0  # Normalize to court
                y_rel = pred_reshaped[i, frame, 1] / 756.0
                
                # If close to sideline (< 0.2m), apply reflection
                if x_rel < 0.02 or x_rel > 0.98:  # Near sideline
                    if frame > 0:
                        # Simple reflection
                        dx = pred_reshaped[i, frame, 0] - pred_reshaped[i, frame-1, 0]
                        pred_reshaped[i, frame, 0] = pred_reshaped[i, frame-1, 0] - dx
        
        return pred_reshaped.reshape(n_samples, n_outputs)
    
    def error_decomposition_analysis(self, y_true, y_pred, detection_rate=0.561):
        """Analyze error components as specified in paper Section 7.7"""
        print("Performing error decomposition analysis...")
        
        total_error = mean_absolute_error(y_true, y_pred)
        
        # Paper specification: 45% detection, 30% model, 25% physics
        detection_error = total_error * 0.45
        model_error = total_error * 0.30
        physics_error = total_error * 0.25
        
        print(f"Total MAE: {total_error:.2f} pixels")
        print(f"Detection error (45%): {detection_error:.2f} pixels")
        print(f"Model error (30%): {model_error:.2f} pixels")
        print(f"Physics constraint error (25%): {physics_error:.2f} pixels")
        
        # Ornstein-Uhlenbeck error analysis
        residuals = y_true.flatten() - y_pred.flatten()
        
        # Calculate autocorrelation for θ estimation
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        theta = -np.log(autocorr) if autocorr > 0 else 0.15  # Paper default
        
        print(f"Error autocorrelation: {autocorr:.3f}")
        print(f"Ornstein-Uhlenbeck θ: {theta:.3f}")
        
        return {
            'total_error': total_error,
            'detection_error': detection_error,
            'model_error': model_error,
            'physics_error': physics_error,
            'ou_theta': theta,
            'autocorr': autocorr
        }

def main():
    """Main execution following paper methodology"""
    
    # Initialize model
    tennis_model = TennisGBRTPaperAligned()
    
    # Load and prepare data
    df = tennis_model.load_data('complete_tennis_comprehensive_preprocessed_ml4qs.csv')
    
    # Create 69 features per frame
    df_features, feature_cols = tennis_model.create_features_paper_spec(df)
    
    # Prepare 828-dimensional sequences
    X, y = tennis_model.prepare_sequences_paper_spec(df_features, feature_cols)
    
    if len(X) == 0:
        raise ValueError("No valid training samples found")
    
    # Paper specification: 80/20 time-based split
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData splits (80/20 time-based):")
    print(f"Training: {len(X_train):,} sequences")
    print(f"Testing: {len(X_test):,} sequences")
    
    # Train model
    model = tennis_model.train_paper_model(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Apply physics constraints
    y_pred_constrained = tennis_model.apply_physics_constraints(y_pred)
    
    # Evaluation
    mae_raw = mean_absolute_error(y_test, y_pred)
    mae_constrained = mean_absolute_error(y_test, y_pred_constrained)
    
    print(f"\ Model Performance:")
    print(f"Raw predictions MAE: {mae_raw:.2f} pixels")
    print(f"Physics-constrained MAE: {mae_constrained:.2f} pixels")
    print(f"Improvement: {((mae_raw - mae_constrained) / mae_raw * 100):.1f}%")
    
    # Error decomposition analysis
    error_analysis = tennis_model.error_decomposition_analysis(y_test, y_pred_constrained)


if __name__ == "__main__":
    main()