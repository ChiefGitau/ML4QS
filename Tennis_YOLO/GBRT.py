import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from scipy.signal import savgol_filter

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ball_center_x'] = (df['ball_x1'] + df['ball_x2']) / 2
    df['ball_center_y'] = (df['ball_y1'] + df['ball_y2']) / 2
    ball_cols = ['ball_center_x', 'ball_center_y']
    df[ball_cols] = df[ball_cols].interpolate(method='linear', limit=5).ffill().bfill()
    if len(df) > 11:
        window_size = min(11, len(df)//2)
        if window_size % 2 == 0: window_size -= 1
        df['ball_center_x'] = savgol_filter(df['ball_center_x'], window_size, 2)
        df['ball_center_y'] = savgol_filter(df['ball_center_y'], window_size, 2)
    return df

def create_features(df, window_size=5):
    time_diff = df['time_seconds'].diff().replace(0, 0.0333)
    df['rel_x'] = df['ball_center_x'] / df['court_width_pixels']
    df['rel_y'] = df['ball_center_y'] / df['court_height_pixels']
    df['vx'] = df['ball_center_x'].diff() / time_diff
    df['vy'] = df['ball_center_y'].diff() / time_diff
    df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
    df['acceleration'] = np.sqrt(df['vx'].diff()**2 + df['vy'].diff()**2) / time_diff
    df['direction_change'] = np.arctan2(df['vy'], df['vx']).diff()
    df['player_dist'] = np.sqrt(
        (df['player_1_center_x'] - df['ball_center_x'])**2 + 
        (df['player_1_center_y'] - df['ball_center_y'])**2
    )
    for col in ['rel_x', 'rel_y', 'vx', 'vy']:
        df[f'{col}_ma3'] = df[col].rolling(3).mean()
        df[f'{col}_ma5'] = df[col].rolling(window_size).mean()
    df['near_net'] = ((df['rel_x'] > 0.4) & (df['rel_x'] < 0.6)).astype(int)
    df['in_corner'] = ((df['rel_y'] < 0.2) | (df['rel_y'] > 0.8)).astype(int)
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
    return df[selected_features + ['ball_center_x', 'ball_center_y']].dropna()

def train_physics_gbrt(X, y):
    tscv = TimeSeriesSplit(n_splits=3, gap=3)
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
    if len(X) > 0:
        sample_weights = np.geomspace(1.0, 3.0, num=len(X))
        opt.fit(X, y, multi_gbrt__sample_weight=sample_weights)
    else:
        opt.fit(X, y)
    return opt.best_estimator_

def main():
    df = load_data('Tennis_YOLO/tennis_comprehensive_temporal_dataset.csv')
    df = create_features(df, window_size=5)
    feature_cols = [col for col in df.columns if col not in ['ball_center_x', 'ball_center_y']]
    hist_frames = 12
    pred_frames = 5
    X, y = [], []
    for i in range(hist_frames, len(df)-pred_frames):
        x_seq = df.iloc[i-hist_frames:i][feature_cols].values.flatten()
        y_seq = df.iloc[i:i+pred_frames][['ball_center_x', 'ball_center_y']].values.flatten()
        if not np.isnan(x_seq).any() and not np.isnan(y_seq).any():
            X.append(x_seq)
            y.append(y_seq)
    X, y = np.array(X), np.array(y)
    print(f"Final training data shape: X={X.shape}, y={y.shape}")
    if len(X) == 0:
        raise ValueError("No valid training samples found. Check data preprocessing.")
    model = train_physics_gbrt(X, y)
    print("\nOptimized model parameters:")
    print(model.named_steps['multi_gbrt'].estimator.get_params())
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"\nTest MAE: {mae:.2f} pixels")

if __name__ == "__main__":
    main()
