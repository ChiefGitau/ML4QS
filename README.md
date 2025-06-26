# Tennis YOLO

Machine learning system for tennis video analysis using YOLO object detection and trajectory prediction.

## Features

- YOLO-based tennis ball and player detection
- LSTM and GBRT models for trajectory prediction
- Video overlay generation with predictions
- Court-aware analysis and measurements
- Comprehensive dataset processing pipeline

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Data Processing
```bash
python combine_tennis_datasets.py
```

### Model Training
- GBRT: Run `tennis_gbrt_preparation_and_training.ipynb`
- LSTM: Run `tennis_lstm_preparation_and_training.ipynb`

### Video Analysis
```bash
python create_tennis_video_overlay.py
```

## Project Structure

- `dataset/` - Tennis tracking data and metadata
- `input_videos/` - Raw tennis videos
- `output_videos/` - Processed videos with overlays
- `models/` - Trained ML models
- `data_analysis_help/` - Analysis utilities
- `*.ipynb` - Jupyter notebooks for training and analysis

## Models

- **LSTM**: Temporal sequence prediction for ball trajectory
- **GBRT**: Gradient boosting for trajectory prediction
- **YOLO**: Object detection for balls and players