# HW1: Autonomous Navigation in Gazebo

## Objective
Implement autonomous navigation of a TurtleBot3 robot in Gazebo Classic using ROS 2 Humble with SLAM and Nav2.

## Setup
- **VM:** Ubuntu 22.04 LTS
- **ROS:** ROS 2 Humble
- **Robot:** TurtleBot3 Burger
- **Simulator:** Gazebo Classic 11.10.2

## How to Run

### Step 1: Launch Gazebo Simulation
```bash
export TURTLEBOT3_MODEL=burger
export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

### Step 2: Launch SLAM (new terminal)
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_cartographer cartographer.launch.py use_sim_time:=True
```

### Step 3: Launch Nav2 (new terminal)
```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_navigation2 navigation2.launch.py use_sim_time:=True
```

### Step 4: Send Navigation Goals
- In RViz2, click **"Navigation2 Goal"**
- Click on the map to set a goal
- Robot navigates autonomously avoiding obstacles

## Results
- Robot successfully builds map using Cartographer SLAM
- Nav2 plans collision-free paths to goals
- Obstacles detected and avoided in real time
- Simulation runs at ~95% Real Time Factor (RTF)
