# HW3: Model Predictive Control (MPC)

## Objective
Implement MPC for trajectory tracking of a TurtleBot3 unicycle robot on a square loop with sharp 90-degree turns.

## How to Run
```bash
python3 mpc_localization.py
```

## Implementation Details
- **Robot Model:** Unicycle (TurtleBot3 Burger)
- **Trajectory:** Square loop with sharp turns
- **Horizon:** 10 steps (1 second)
- **Optimization:** Random shooting (500 samples)
- **Constraints:** v ∈ [-0.5, 1.0] m/s, ω ∈ [-1.0, 1.0] rad/s

## Generated Outputs
- `mpc_trajectory.png` — Reference vs MPC tracked trajectory
- `mpc_error.png` — Position tracking error over time
- `mpc_controls.png` — Linear and angular velocity inputs
- `mpc_animation.avi` — Animation video (play with VLC)

## Create Video
```bash
python3 mpc_localization.py
ffmpeg -y -framerate 20 -i "mpc_frames/frame_%04d.png" -vcodec mpeg4 -q:v 2 mpc_animation.avi
```
