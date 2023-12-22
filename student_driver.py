#!/usr/bin/env python3

import sys
import rospy
from new_driver import Driver
from math import atan2, sqrt, pi

class StudentDriver(Driver):
    '''
    StudentDriver extends the Driver class to provide custom navigation logic.
    It controls the robot's movement towards specified waypoints while avoiding obstacles.
    '''

    def __init__(self, threshold=0.6):
        '''
        Initialize the StudentDriver.
        - threshold: Distance within which the robot considers it has reached a waypoint.
        '''
        super().__init__('odom')
        self._threshold = threshold  # Distance threshold for waypoint proximity.

    def close_enough_to_waypoint(self, distance, target, lidar):
        '''
        Determine if the robot is close enough to the current waypoint.
        - distance: Distance to the current waypoint.
        - target: Target waypoint coordinates.
        - lidar: LIDAR sensor data.
        Returns True if close enough, False otherwise.
        '''
        return distance < self._threshold

    def get_twist(self, target, lidar):
        '''
        Calculate the twist command to navigate the robot.
        - target: Target waypoint coordinates.
        - lidar: LIDAR sensor data.
        Returns a Twist message to control the robot.
        '''
        angle_to_target = atan2(target[1], target[0])  # Angle to the waypoint.
        distance_to_target = sqrt(target[0] ** 2 + target[1] ** 2)  # Distance to the waypoint.

        # Default twist command (stop).
        command = Driver.zero_twist()

        # Adjust linear velocity based on distance to obstacle and target.
        min_distance_to_obstacle = min(lidar.ranges)  # Closest obstacle distance.
        command.linear.x = min(0.5, 0.1 * distance_to_target) if min_distance_to_obstacle > 0.3 else -0.5

        # Adjust angular velocity based on obstacle presence and target angle.
        angle_of_closest_obstacle = lidar.ranges.index(min_distance_to_obstacle) * lidar.angle_increment
        if min_distance_to_obstacle < 0.3:  # If too close to an obstacle.
            command.angular.z = -0.5 if angle_of_closest_obstacle < pi else 0.5
        else:
            command.angular.z = 2.0 * angle_to_target

        # Check if the robot is close enough to the waypoint.
        if self.close_enough_to_waypoint(distance_to_target, target, lidar):
            self._target_point = None  # Reset target point for the next waypoint.

        rospy.loginfo(f"Angle to Target: {angle_to_target:.2f}, Distance to Target: {distance_to_target:.2f}, " +
                      f"Closest Obstacle Angle: {angle_of_closest_obstacle:.2f}, Distance: {min_distance_to_obstacle:.2f}")

        return command

if __name__ == '__main__':
    rospy.init_node('student_driver', argv=sys.argv)
    driver = StudentDriver()
    rospy.spin()  # Keep the node running.
