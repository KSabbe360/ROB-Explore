# Autonomous Navigation Robot Project

## Overview
This project demonstrates an **autonomous navigation system** for a robotic platform using ROS (Robot Operating System). The system enables the robot to autonomously navigate through an environment, intelligently reaching waypoints and avoiding obstacles.

### Components

#### `student_driver.py`
- **Purpose**: Controls the robot's movement towards waypoints.
- **Functionality**: 
  - **Waypoint Proximity Detection**: Determines if the robot is close enough to a waypoint.
  - **Obstacle Avoidance**: Modifies the path when obstacles are detected within a certain threshold.
  - **Angular and Linear Velocity Adjustment**: Adjusts the robot's speed and direction based on the distance to the target and obstacle proximity.

#### `student_controller.py`
- **Purpose**: Manages high-level navigation tasks and waypoint generation.
- **Functionality**: 
  - **Dynamic Path Planning**: Generates new waypoints based on the robot's current position and unexplored map areas.
  - **Integration with ROS**: Utilizes ROS messages and services for map data handling and path computation.
  - **Reactive Navigation**: Adjusts waypoints in response to changes in the environment or robot's position.

#### `exploring.py`
- **Purpose**: Handles the exploration strategy and waypoint generation.
- **Functionality**: 
  - **Exploration Point Identification**: Locates unexplored areas that are accessible based on the current map data.
  - **Coordinate Conversion**: Converts between real-world coordinates and pixel values on the map.
  - **Waypoint Generation**: Creates a series of waypoints along a planned path for the robot to follow.

#### `path_planning.py`
- **Purpose**: Implements path planning algorithms.
- **Functionality**: 
  - **Pathfinding Algorithm**: Implements a grid-based Dijkstra's algorithm to find the shortest path between points on the map.
  - **Heuristic Function for A***: Enhances pathfinding efficiency with a heuristic function for A* implementation.
  - **Map Interaction**: Works with thresholded map images to identify free, occupied, and unseen areas for path planning.

#### `new_driver.py`
- **Purpose**: Base driver class for fundamental driving functionalities.
- **Functionality**: 
  - **Action Server Interface**: Integrates with ROS action servers to receive and respond to navigation goals.
  - **Basic Movement Commands**: Provides fundamental driving commands for the robot.
  - **LIDAR Data Processing**: Subscribes to LIDAR data for obstacle detection and avoidance.
