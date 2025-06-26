import pandas as pd

# Add court region classification with detailed zones
def classify_detailed_court_region(x_rel, y_rel, player_id):
    if pd.isna(x_rel) or pd.isna(y_rel):
        return 'unknown'

    # Determine side of court based on Y position
    if y_rel < 0.5:
        court_half = "far_half"
    else:
        court_half = "near_half"

    # Determine court zone based on Y position
    if y_rel < 0.2 or y_rel > 0.8:
        zone = "baseline_area"
    elif y_rel < 0.35 or y_rel > 0.65:
        zone = "service_area"
    else:
        zone = "net_area"

    # Determine left/center/right based on X position
    if x_rel < 0.33:
        side = "left"
    elif x_rel > 0.67:
        side = "right"
    else:
        side = "center"

    return f"{court_half}_{zone}_{side}"

    # Ball court region classification
def classify_ball_court_region(x_rel, y_rel):
    if pd.isna(x_rel) or pd.isna(y_rel):
        return 'unknown'

    if y_rel < 0.1:
        return "behind_far_baseline"
    elif y_rel < 0.35:
        return "far_court"
    elif y_rel < 0.45:
        return "far_service_area"
    elif y_rel < 0.55:
        return "net_area"
    elif y_rel < 0.65:
        return "near_service_area"
    elif y_rel < 0.9:
        return "near_court"
    else:
        return "behind_near_baseline"