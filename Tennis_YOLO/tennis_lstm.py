"""
Tennis Ball Trajectory LSTM Prediction

- Two LSTM layers (128 and 64 units)
- Input: 12 frames × 69 features
- Output: 5 future frames × 2 coordinates
- Bayesian hyperparameter optimization
- Temporal evaluation with error analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterSampler
import warnings
warnings.filterwarnings('ignore')

class TennisLSTMDataset(Dataset):
    """PyTorch dataset for tennis trajectory sequences"""
    
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TennisLSTMModel(nn.Module):

    
    def __init__(self, input_size=69, lstm1_units=128, lstm2_units=64, dropout_rate=0.2):
        super(TennisLSTMModel, self).__init__()
        

        self.lstm1 = nn.LSTM(input_size, lstm1_units, batch_first=True, dropout=dropout_rate)
        self.lstm2 = nn.LSTM(lstm1_units, lstm2_units, batch_first=True, dropout=dropout_rate)
        
        # Dense layer with ReLU activation
        self.dense = nn.Linear(lstm2_units, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        
        # TimeDistributed output layer for 5 frames × 2 coordinates
        self.output_layer = nn.Linear(64, 10)  # 5 frames × 2 coordinates = 10 outputs
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length=12, features=69)
        
        # First LSTM layer (128 units)
        h1, _ = self.lstm1(x)
        
        # Second LSTM layer (64 units)
        h2, _ = self.lstm2(h1)
        
        # Use the last timestep output
        h2_last = h2[:, -1, :]  # (batch_size, 64)
        
        # Dense layer with ReLU
        dense_out = self.relu(self.dense(h2_last))
        dense_out = self.dropout(dense_out)
        
        # Output layer (TimeDistributed equivalent)
        output = self.output_layer(dense_out)  # (batch_size, 10)
        
        # Reshape to (batch_size, 5, 2) for 5 frames × 2 coordinates
        output = output.view(-1, 5, 2)
        
        return output

class TennisLSTMAligned:
    """
    LSTM model aligned with specifications for tennis ball trajectory prediction
    """
    
    def __init__(self):
        self.sequence_length = 12  # Input frames
        self.pred_frames = 5      # Output frames
        self.features_per_frame = 69
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        
        print(f"Using device: {self.device}")
    
    def load_and_prepare_data(self, filepath):
        """Load and prepare data"""
        print("Loading data for LSTM ")
        
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
                    df['time_seconds'] = np.arange(len(df)) / 30.0
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        # Basic interpolation
        ball_cols = ['ball_center_x', 'ball_center_y']
        df[ball_cols] = df[ball_cols].interpolate(method='linear', limit=5).ffill().bfill()
        
        # Savitzky-Golay filtering (order=2, window=11)
        if len(df) > 11:
            try:
                df['ball_center_x'] = savgol_filter(df['ball_center_x'], 11, 2)
                df['ball_center_y'] = savgol_filter(df['ball_center_y'], 11, 2)
                print("Applied Savitzky-Golay filtering (window=11, order=2)")
            except Exception as e:
                print(f"Savitzky-Golay filtering failed: {e}")
        
        return df
    
    def create_lstm_features(self, df):
        """Create 69 features per frame for LSTM (same as GBRT)"""
        print(" Creating 69 features per frame for LSTM...")
        
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
        
        # Create the same 69 features as GBRT model
        # Court-relative positions (2 features)
        df_features['rel_x'] = df_features['ball_center_x'] / df_features['court_width_pixels']
        df_features['rel_y'] = df_features['ball_center_y'] / df_features['court_height_pixels']
        
        # Velocity components (4 features)
        df_features['vx'] = df_features['ball_center_x'].diff() / time_diff
        df_features['vy'] = df_features['ball_center_y'].diff() / time_diff
        
        # Convert to m/s
        pixels_per_meter = 100
        df_features['vx_ms'] = df_features['vx'] / pixels_per_meter
        df_features['vy_ms'] = df_features['vy'] / pixels_per_meter
        
        # Speed and acceleration (3 features)
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
        
        # Create all 69 features
        base_vars = ['rel_x', 'rel_y', 'vx', 'vy', 'speed', 'acceleration', 'direction_change', 'player_dist']
        
        # Moving averages (16 features: 8 vars × 2 windows)
        for var in base_vars:
            df_features[f'{var}_ma3'] = df_features[var].rolling(3, min_periods=1).mean()
            df_features[f'{var}_ma5'] = df_features[var].rolling(5, min_periods=1).mean()
        
        # Standard deviations (8 features)
        for var in base_vars:
            df_features[f'{var}_std5'] = df_features[var].rolling(5, min_periods=1).std()
        
        # Court zone features (2 features)
        df_features['near_net'] = ((df_features['rel_x'] > 0.4) & (df_features['rel_x'] < 0.6)).astype(int)
        df_features['in_corner'] = ((df_features['rel_y'] < 0.2) | (df_features['rel_y'] > 0.8)).astype(int)
        
        # Serve/rally state (1 feature)
        estimated_height = 1.5
        df_features['serve_rally_state'] = (
            (df_features['speed_ms'] > 15) & (estimated_height < 2)
        ).astype(int)
        
        # Additional features to reach 69 total
        df_features['trajectory_curvature'] = np.abs(df_features['direction_change'])
        df_features['vx_vy_ratio'] = np.abs(df_features['vx'] / (df_features['vy'] + 1e-6))
        df_features['speed_change'] = df_features['speed'].diff()
        df_features['distance_to_net'] = np.abs(df_features['rel_x'] - 0.5)
        df_features['distance_to_baseline'] = np.minimum(df_features['rel_y'], 1 - df_features['rel_y'])
        df_features['distance_to_sideline'] = np.minimum(df_features['rel_x'], 1 - df_features['rel_x'])
        df_features['frame_number'] = np.arange(len(df_features))
        df_features['time_in_rally'] = df_features['time_seconds'] - df_features['time_seconds'].iloc[0]
        df_features['rel_x_smooth'] = df_features['rel_x'].rolling(7, min_periods=1).mean()
        df_features['rel_y_smooth'] = df_features['rel_y'].rolling(7, min_periods=1).mean()
        df_features['vx_smooth'] = df_features['vx'].rolling(5, min_periods=1).mean()
        df_features['vy_smooth'] = df_features['vy'].rolling(5, min_periods=1).mean()
        df_features['speed_smooth'] = df_features['speed'].rolling(5, min_periods=1).mean()
        df_features['kinetic_energy'] = 0.5 * df_features['speed_ms']**2
        df_features['speed_max5'] = df_features['speed'].rolling(5, min_periods=1).max()
        df_features['speed_min5'] = df_features['speed'].rolling(5, min_periods=1).min()
        
        # Select exactly 69 features
        feature_list = [
            'rel_x', 'rel_y', 'vx', 'vy', 'vx_ms', 'vy_ms',
            'speed', 'speed_ms', 'acceleration', 'direction_change', 'speed_change',
            'player_dist', 'trajectory_curvature', 'vx_vy_ratio',
            'distance_to_net', 'distance_to_baseline', 'distance_to_sideline',
            'frame_number', 'time_in_rally',
            'rel_x_smooth', 'rel_y_smooth', 'vx_smooth', 'vy_smooth', 'speed_smooth',
            'kinetic_energy', 'speed_max5', 'speed_min5',
            'near_net', 'in_corner', 'serve_rally_state',
            'court_width_pixels', 'court_height_pixels'
        ]
        
        # Add moving averages
        for var in base_vars:
            feature_list.extend([f'{var}_ma3', f'{var}_ma5', f'{var}_std5'])

        feature_list = feature_list[:69]

        target_features = ['ball_center_x', 'ball_center_y']
        all_features = feature_list + target_features
        
        # Filter and clean
        available_features = [col for col in all_features if col in df_features.columns]
        result_df = df_features[available_features].dropna()
        
        print(f"Created {len(feature_list)} features per frame")
        print(f"Dataset shape: {result_df.shape}")
        
        return result_df, feature_list
    
    def prepare_lstm_sequences(self, df, feature_cols):
        """Prepare sequences for LSTM: (batch, seq_len=12, features=69)"""
        print(f"Preparing LSTM sequences: ({self.sequence_length}, {len(feature_cols)}) -> (5, 2)")
        
        target_cols = ['ball_center_x', 'ball_center_y']
        
        X, y = [], []
        for i in range(self.sequence_length, len(df) - self.pred_frames + 1):
            # Input sequence: 12 frames × 69 features
            x_seq = df.iloc[i-self.sequence_length:i][feature_cols].values  # Shape: (12, 69)
            
            # Target sequence: 5 frames × 2 coordinates
            y_seq = df.iloc[i:i+self.pred_frames][target_cols].values  # Shape: (5, 2)
            
            if not np.isnan(x_seq).any() and not np.isnan(y_seq).any():
                X.append(x_seq)
                y.append(y_seq)
        
        X, y = np.array(X), np.array(y)
        
        print(f" Input shape: {X.shape} (batch, seq_len, features)")
        print(f" Output shape: {y.shape} (batch, pred_frames, coordinates)")
        print(f"  Total sequences: {len(X)}")
        
        return X, y
    
    def preprocess_for_lstm(self, X_train, X_test, y_train, y_test):
        """Apply Z-score normalization """
        print("🔧 Applying Z-score normalization...")
        
        # Reshape for normalization: (batch*seq_len, features)
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        # Fit scaler on training data only
        self.scaler.fit(X_train_reshaped)
        
        # Transform both sets
        X_train_scaled = self.scaler.transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        # Reshape back to sequences
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        print(f"Normalized input features")
        print(f" Training mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def create_model(self, lstm1_units=128, lstm2_units=64, dropout_rate=0.2):
        """Create LSTM model"""
        model = TennisLSTMModel(
            input_size=self.features_per_frame,
            lstm1_units=lstm1_units,
            lstm2_units=lstm2_units,
            dropout_rate=dropout_rate
        )
        return model.to(self.device)
    
    def train_model(self, model, train_loader, val_loader, learning_rate=0.001, epochs=100, patience=10):
        """Train LSTM model with early stopping"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        

        print(f"   Epochs: {epochs}, Patience: {patience}")
        print(f"   Learning rate: {learning_rate}")
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1:3d}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save(model.state_dict(), 'best_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break
        

        model.load_state_dict(torch.load('best_lstm_model.pth'))
        
        return model, train_losses, val_losses
    
    def bayesian_optimization(self, X_train, X_test, y_train, y_test):
        """Bayesian hyperparameter optimization """
        print("Performing Bayesian hyperparameter optimization...")

        param_space = {
            'lstm1_units': [64, 128, 256],
            'lstm2_units': [32, 64, 128],
            'dropout_rate': np.linspace(0.0, 0.3, 10),
            'learning_rate': np.logspace(-3.3, -2.3, 20),  # [0.0005, 0.005]
            'batch_size': [32, 64]
        }
        
        # Create parameter samples for optimization (30 iterations)
        param_list = list(ParameterSampler(param_space, n_iter=30, random_state=42))
        
        best_score = float('inf')
        best_params = None
        results = []
        
        print(f" Testing {len(param_list)} parameter combinations")
        
        for i, params in enumerate(param_list):
            print(f"Iteration {i+1}/30: {params}")
            
            try:
                # Create model with current parameters
                model = self.create_model(
                    lstm1_units=params['lstm1_units'],
                    lstm2_units=params['lstm2_units'],
                    dropout_rate=params['dropout_rate']
                )
                
                # Create data loaders
                train_dataset = TennisLSTMDataset(X_train, y_train)
                val_dataset = TennisLSTMDataset(X_test, y_test)
                
                train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
                
                # Train model (reduced epochs for optimization)
                model, _, _ = self.train_model(
                    model, train_loader, val_loader,
                    learning_rate=params['learning_rate'],
                    epochs=50,  # Reduced for optimization
                    patience=5
                )
                
                # Evaluate
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = model(batch_x)
                        loss = nn.MSELoss()(outputs, batch_y)
                        val_loss += loss.item()
                
                val_loss /= len(val_loader)
                
                results.append({
                    'params': params,
                    'score': val_loss
                })
                
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                    print(f"New best score: {val_loss:.6f}")
                
            except Exception as e:
                print(f"Failed: {e}")
                continue
        
        print(f"Best parameters: {best_params}")
        print(f"Best score: {best_score:.6f}")
        
        return best_params, results
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model performance """

        
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                outputs = model(batch_x)
                
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(batch_y.cpu().numpy())
        
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Calculate RMSE for X and Y coordinates
        x_predictions = predictions[:, :, 0].flatten()
        y_predictions = predictions[:, :, 1].flatten()
        x_targets = targets[:, :, 0].flatten()
        y_targets = targets[:, :, 1].flatten()
        
        # Convert to meters (assuming 100 pixels per meter)
        pixels_per_meter = 100
        
        rmse_x = np.sqrt(mean_squared_error(x_targets, x_predictions)) / pixels_per_meter
        rmse_y = np.sqrt(mean_squared_error(y_targets, y_predictions)) / pixels_per_meter
        
        mae_x = mean_absolute_error(x_targets, x_predictions) / pixels_per_meter
        mae_y = mean_absolute_error(y_targets, y_predictions) / pixels_per_meter
        
        print(f" RMSE X-axis: {rmse_x:.3f} meters")
        print(f" RMSE Y-axis: {rmse_y:.3f} meters")
        print(f" MAE X-axis: {mae_x:.3f} meters")
        print(f" MAE Y-axis: {mae_y:.3f} meters")

        
        return {
            'rmse_x': rmse_x,
            'rmse_y': rmse_y,
            'mae_x': mae_x,
            'mae_y': mae_y,
            'predictions': predictions,
            'targets': targets
        }
    
    def error_analysis(self, predictions, targets, serve_rally_states=None):

        print("error analysis...")
        
        # Reshape for analysis
        pred_flat = predictions.reshape(-1, 2)
        target_flat = targets.reshape(-1, 2)
        
        # Overall error
        rmse_overall = np.sqrt(mean_squared_error(target_flat, pred_flat))
        
        # Error by prediction horizon (5 frames)
        horizon_errors = []
        for frame in range(5):
            frame_pred = predictions[:, frame, :]
            frame_target = targets[:, frame, :]
            frame_rmse = np.sqrt(mean_squared_error(frame_target, frame_pred))
            horizon_errors.append(frame_rmse)
        
        print(f" Error by prediction horizon:")
        for i, error in enumerate(horizon_errors):
            print(f"Frame {i+1}: {error:.3f} pixels")
        

        error_increase = (horizon_errors[-1] - horizon_errors[0]) / horizon_errors[0] * 100
        print(f"Error increase first->last frame: {error_increase:.1f}%")

        
        # Spatial error analysis
        x_errors = np.abs(pred_flat[:, 0] - target_flat[:, 0])
        y_errors = np.abs(pred_flat[:, 1] - target_flat[:, 1])
        
        return {
            'rmse_overall': rmse_overall,
            'horizon_errors': horizon_errors,
            'error_increase_pct': error_increase,
            'x_errors': x_errors,
            'y_errors': y_errors
        }
    
    def create_spatial_heatmap(self, predictions, targets, court_width=1344, court_height=756):

        print("Creating spatial error heatmap")
        
        pred_flat = predictions.reshape(-1, 2)
        target_flat = targets.reshape(-1, 2)
        
        errors = np.sqrt(np.sum((pred_flat - target_flat)**2, axis=1))
        
        # Create 2D histogram of errors
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Error heatmap
        hist, xedges, yedges = np.histogram2d(
            target_flat[:, 0], target_flat[:, 1], 
            bins=20, weights=errors
        )
        counts, _, _ = np.histogram2d(target_flat[:, 0], target_flat[:, 1], bins=20)
        
        # Avoid division by zero
        avg_errors = np.divide(hist, counts, out=np.zeros_like(hist), where=counts!=0)
        
        im1 = ax1.imshow(avg_errors.T, origin='lower', cmap='Reds', 
                        extent=[0, court_width, 0, court_height])
        ax1.set_title('Spatial Error Heatmap\n(Average Prediction Error)')
        ax1.set_xlabel('Court X (pixels)')
        ax1.set_ylabel('Court Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Average Error (pixels)')
        
        # Prediction density
        im2 = ax2.hist2d(target_flat[:, 0], target_flat[:, 1], bins=20, cmap='Blues')
        ax2.set_title('Prediction Density\n(Number of Predictions)')
        ax2.set_xlabel('Court X (pixels)')
        ax2.set_ylabel('Court Y (pixels)')
        plt.colorbar(im2[3], ax=ax2, label='Count')
        
        plt.tight_layout()
        plt.savefig('lstm_spatial_error_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return avg_errors

def main():

    print("TENNIS BALL TRAJECTORY LSTM PREDICTION")
    print("=" * 65)
    
    # Initialize model
    lstm_model = TennisLSTMAligned()
    
    # Load and prepare data
    df = lstm_model.load_and_prepare_data('complete_tennis_comprehensive_preprocessed_ml4qs.csv')
    
    # Create features
    df_features, feature_cols = lstm_model.create_lstm_features(df)
    
    # Prepare sequences
    X, y = lstm_model.prepare_lstm_sequences(df_features, feature_cols)
    
    if len(X) == 0:
        raise ValueError("No valid training samples found")

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"\nData splits (80/20 temporal):")
    print(f"Training: {len(X_train):,} sequences")
    print(f"Testing: {len(X_test):,} sequences")
    
    # Preprocessing
    X_train_norm, X_test_norm, y_train_norm, y_test_norm = lstm_model.preprocess_for_lstm(
        X_train, X_test, y_train, y_test
    )
    
    # Bayesian hyperparameter optimization
    best_params, opt_results = lstm_model.bayesian_optimization(
        X_train_norm, X_test_norm, y_train_norm, y_test_norm
    )
    
    # Train final model with best parameter
    final_model = lstm_model.create_model(
        lstm1_units=best_params['lstm1_units'],
        lstm2_units=best_params['lstm2_units'],
        dropout_rate=best_params['dropout_rate']
    )
    
    # Create data loaders
    train_dataset = TennisLSTMDataset(X_train_norm, y_train_norm)
    test_dataset = TennisLSTMDataset(X_test_norm, y_test_norm)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)
    
    # Train final model
    final_model, train_losses, val_losses = lstm_model.train_model(
        final_model, train_loader, test_loader,
        learning_rate=best_params['learning_rate'],
        epochs=200,
        patience=10
    )
    
    # Evaluation
    results = lstm_model.evaluate_model(final_model, test_loader)
    
    # Error analysis
    error_analysis = lstm_model.error_analysis(results['predictions'], results['targets'])
    
    # Create spatial heatmap
    spatial_errors = lstm_model.create_spatial_heatmap(results['predictions'], results['targets'])
    
    # Final summary
    print(f"Final Performance:")
    print(f"RMSE X-axis: {results['rmse_x']:.3f} meters (target: 1.19m)")
    print(f"RMSE Y-axis: {results['rmse_y']:.3f} meters (target: 1.04m)")
    print(f"Error horizon increase: {error_analysis['error_increase_pct']:.1f}% (target: 26%)")
    
    print(f"\nModel Architecture:")
    print(f"LSTM Layer 1: {best_params['lstm1_units']} units")
    print(f"LSTM Layer 2: {best_params['lstm2_units']} units")
    print(f"Dropout: {best_params['dropout_rate']:.3f}")
    print(f"Learning rate: {best_params['learning_rate']:.4f}")
    print(f"Batch size: {best_params['batch_size']}")
    
    # Save final model
    torch.save(final_model.state_dict(), 'models/tennis_lstm_final_model.pth')
    print(f"\ Model saved: tennis_lstm_final_model.pth")

if __name__ == "__main__":
    main()