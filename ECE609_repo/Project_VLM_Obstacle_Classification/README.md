# Course Project: VLM-Based Real-Time Obstacle Classification for Autonomous Robot Navigation

## Team
- Vandan Patel
- Joel V George

## Overview
This project integrates a lightweight Vision-Language Model (VLM) into the Nav2 navigation pipeline of a TurtleBot3 robot to enable semantic obstacle classification and adaptive navigation behavior.

## Problem Statement
Can a small, computationally efficient VLM provide accurate real-time semantic obstacle classification to improve autonomous robot navigation in a ROS 2 simulation environment?

## Approach
1. **Navigation Pipeline:** TurtleBot3 in Gazebo Classic with Nav2 + Cartographer SLAM
2. **VLM Integration:** CLIP/MobileVLM running as ROS 2 node, subscribing to `/oakd/rgb/preview/image_raw`
3. **Behavior Adaptation:** Navigation parameters adjusted based on obstacle classification

## Timeline
- **Weeks 3-4:** VLM model selection, environment setup, baseline navigation
- **Weeks 5-6:** VLM integration into ROS 2, initial classification results
- **Weeks 7-8:** Behavior adaptation, benchmarking different VLMs
- **Weeks 9-10:** Final evaluation, report, presentation

## Tools
- ROS 2 Humble
- Gazebo Classic 11.10.2
- TurtleBot3 Burger
- CLIP / MobileVLM
- Nav2 + Cartographer SLAM
- Ubuntu 22.04 LTS
