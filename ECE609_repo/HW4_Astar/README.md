# HW4: A* Path Planning

## Objective
Implement A* algorithm to find a collision-free path from start to goal on a 20x20 grid with obstacles, then simulate the robot following the path.

## How to Run
```bash
python3 astar.py
```

## Implementation Details
- **Grid:** 20x20 with obstacles arranged as walls
- **Start:** (1, 1) — bottom left
- **Goal:** (18, 18) — top right
- **Connectivity:** 8-connected grid
- **Heuristic:** Euclidean distance
- **Robot Simulation:** Smooth interpolation along A* path

## Generated Outputs
- `astar_result.png` — Grid with explored nodes + optimal path
- `astar_path.png` — Clean path visualization
- `astar_cost.png` — Cumulative path cost over steps
- `astar_animation.avi` — Robot following the path (play with VLC)

## Create Video
```bash
python3 astar.py
ffmpeg -y -framerate 20 -i "astar_frames/frame_%04d.png" -vcodec mpeg4 -q:v 2 astar_animation.avi
```
