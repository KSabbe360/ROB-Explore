#!/usr/bin/env python3

import sys
import rospy
import signal
from lab2.msg import NavTargetAction, NavTargetActionGoal
import path_planning
import exploring
import numpy as np

from controller import RobotController

def occupancy_grid_to_numpy(occupancy_grid):
    """
    Convert an OccupancyGrid message to a 2D NumPy array.
    - occupancy_grid: The OccupancyGrid message from ROS.
    Returns a 2D NumPy array representing the occupancy grid.
    """
    data = np.array(occupancy_grid.data)
    width = occupancy_grid.info.width
    height = occupancy_grid.info.height
    return np.reshape(data, (height, width))
   
class StudentController(RobotController):
    """
    StudentController extends RobotController for custom behavior.
    Handles navigation and pathfinding for the robot.
    """

    def __init__(self):
        """
        Initialize the StudentController.
        """
        super().__init__()
        self.previous_distance = None
        self.current_position = None

    def distance_update(self, distance):
        """
        Handle updates in distance to the current goal.
        - distance: Current distance to the goal.
        """
        rospy.loginfo(f'Distance: {distance}')

        # Recalculate path if not making progress towards goal
        if self.previous_distance is not None and distance >= self.previous_distance:
            rospy.loginfo("Robot not making expected progress towards goal. Recalculating path...")
            if self.current_position is not None:
                # Convert current position to pixel coordinates and find a new path
                current_pix = exploring.convert_x_y_to_pix((self._map_data.width, self._map_data.height), self.current_position, self._map_data.resolution)
                new_goal = exploring.find_best_point(self.map, exploring.find_all_possible_goals(self.map), current_pix)
                new_path = path_planning.dijkstra(self.map, current_pix, new_goal)
                self.set_waypoints(new_path if new_path else [])
            else:
                rospy.loginfo("Current position not available")

            self.previous_distance = distance

    def map_update(self, point, map, map_data):
        """
        Handle map updates.
        - point: Current robot position.
        - map: Occupancy grid map.
        - map_data: Data about the map.
        """
        rospy.loginfo('Got a map update.')
        try:
            # Convert the map to a NumPy array and find a new path
            map_array = occupancy_grid_to_numpy(map)
            robot_position = (point.point.x, point.point.y)
            robot_pix = exploring.convert_x_y_to_pix((map_data.width, map_data.height), robot_position, map_data.resolution)
            path = path_planning.dijkstra(map_array, robot_pix, exploring.find_best_point(map_array, exploring.find_all_possible_goals(map_array), robot_pix))
            waypoints = exploring.find_waypoints(map_array, path)
            waypoints_real = [exploring.convert_pix_to_x_y((map_data.width, map_data.height), wp, map_data.resolution) for wp in waypoints]
            self.set_waypoints(waypoints_real)
        except Exception as e:
            rospy.loginfo(f'Error during map update: {e}')

if __name__ == '__main__':
    # Initialize the ROS node and start the controller
    rospy.init_node('student_controller', argv=sys.argv)
    controller = StudentController()
    controller.set_waypoints(((-4, -3), (-4, 0), (5, 0)))
    controller.send_points()
    rospy.spin()
