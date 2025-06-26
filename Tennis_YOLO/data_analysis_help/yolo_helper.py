

from Tennis_YOLO.tennis_utils import PlayerTracker, BallTracker, CourtLineDetector


def initialize_yolo_trackers():
    print(" INITIALIZING YOLO TRACKERS FOR BATCH PROCESSING")

    try:
        player_tracker = PlayerTracker(model_path='yolov8x')
        print(" Player tracker initialized successfully")
    except Exception as e:
        print(f" Error initializing player tracker: {e}")
        player_tracker = None

    try:
        ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
        print("Ball tracker initialized successfully")
    except Exception as e:
        print(f"Error initializing ball tracker: {e}")
        ball_tracker = None

    try:
        court_detector = CourtLineDetector(model_path='models/keypoints_model.pth')
        print(" Court line detector initialized successfully")
    except Exception as e:
        print(f"Error initializing court detector: {e}")
        court_detector = None

    # Check if we can proceed with processing
    if all([player_tracker, ball_tracker, court_detector]):
        print("\nAll trackers ready for batch processing!")
        trackers_ready = True
    else:
        print("\n Some trackers failed to initialize. Will use stub data where available.")
        trackers_ready = False

    print(f"\nBatch processing mode: {'Live detection' if trackers_ready else 'Stub data fallback'}")
    try:
        player_tracker = PlayerTracker(model_path='yolov8x')
        print(" Player tracker initialized successfully")
    except Exception as e:
        print(f" Error initializing player tracker: {e}")
        player_tracker = None

    try:
        ball_tracker = BallTracker(model_path='models/yolo5_last.pt')
        print("Ball tracker initialized successfully")
    except Exception as e:
        print(f"Error initializing ball tracker: {e}")
        ball_tracker = None

    try:
        court_detector = CourtLineDetector(model_path='models/keypoints_model.pth')
        print(" Court line detector initialized successfully")
    except Exception as e:
        print(f"Error initializing court detector: {e}")
        court_detector = None

    # Check if we can proceed with processing
    if all([player_tracker, ball_tracker, court_detector]):
        print("\nAll trackers ready for batch processing!")
        trackers_ready = True
    else:
        print("\n Some trackers failed to initialize. Will use stub data where available.")
        trackers_ready = False


    return trackers_ready, player_tracker, ball_tracker, court_detector

    print(f"\nBatch processing mode: {'Live detection' if trackers_ready else 'Stub data fallback'}")