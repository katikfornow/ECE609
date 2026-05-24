# HW2: EKF Localization

## Objective
Implement Extended Kalman Filter (EKF) localization for a mobile robot with unicycle motion model and range-and-bearing observation model.

## How to Run
```bash
python3 ekf_localization.py
```

## Implementation Details
- **Motion Model:** Unicycle (differential drive)
- **Observation Model:** Range and bearing to 8 landmarks
- **Trajectory:** Circular path
- **Outputs:** 3 plots + animation video

## Generated Outputs
- `ekf_trajectory.png` — True, predicted, EKF corrected trajectories + uncertainty ellipses
- `ekf_error.png` — Position error: Dead reckoning vs EKF
- `ekf_uncertainty.png` — EKF covariance trace over time
- `ekf_animation.avi` — Animation video (play with VLC)
