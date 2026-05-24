# HW2_2: Monte Carlo Localization (Particle Filter)

## Objective
Implement MCL (Particle Filter) for robot localization using 1000 particles with unicycle motion model and range-and-bearing observation model.

## How to Run
```bash
python3 mcl_localization.py
```

## Implementation Details
- **Motion Model:** Unicycle with Gaussian noise
- **Observation Model:** Range and bearing to 8 landmarks
- **Particles:** 1000
- **Resampling:** Low-variance resampling
- **Trajectory:** Circular path

## Generated Outputs
- `mcl_trajectory.png` — True path, MCL estimate, particle clouds
- `mcl_error.png` — MCL position error over time
- `mcl_spread.png` — Particle spread over time
- `mcl_animation.avi` — Animation video (play with VLC)

## Create Video
```bash
python3 mcl_localization.py
ffmpeg -y -framerate 20 -i "mcl_frames/frame_%04d.png" -vcodec mpeg4 -q:v 2 mcl_animation.avi
```
