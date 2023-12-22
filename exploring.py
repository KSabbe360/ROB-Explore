#!/usr/bin/env python3

import numpy as np
import path_planning as path_planning
import imageio
import rospy
import matplotlib.pyplot as plt

def plot_with_explore_points(im_threshhold, zoom=1.0, robot_loc=None, explore_points=None, best_pt=None):
    """
    Visualize exploration points on the map. This function creates two plots: the original and the thresholded
    images of the SLAM map, highlighting the robot's location, exploration points, and the best point for exploration.
    """
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[1].set_title("Threshold Image")

    # Plot exploration points on the thresholded image
    if explore_points is not None:
        for p in explore_points:
            axs[1].plot(p[0], p[1], '.b', markersize=2)

    # Highlight the robot's location and the best exploration point
    for i in range(2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if best_pt is not None:
            axs[i].plot(best_pt[0], best_pt[1], '*y', markersize=10)
        axs[i].axis('equal')

    # Apply zooming to focus on a specific area of the map
    for i in range(2):
        width = im_threshhold.shape[1]
        height = im_threshhold.shape[0]
        axs[i].set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        axs[i].set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)

def convert_pix_to_x_y(im_size, pix, size_pix):
    """
    Convert pixel coordinates to real-world coordinates, assuming the center of the image is the origin (0,0).
    Checks if the provided pixel coordinates are within the bounds of the image.
    """
    if not (0 <= pix[0] < im_size[0]) or not (0 <= pix[1] < im_size[1]):
        raise ValueError(f"Pixel {pix} not in image, image size {im_size}")

    x_real = (pix[0] - im_size[0] / 2.0) * size_pix
    y_real = (pix[1] - im_size[1] / 2.0) * size_pix
    return x_real, y_real

def convert_x_y_to_pix(im_size, x_y, size_pix):
    """
    Convert real-world coordinates to pixel coordinates, assuming the center of the image is the origin (0,0).
    Adjusts the computed pixel coordinates to ensure they are within the bounds of the image.
    """
    pix_x = int((x_y[0] / size_pix) + (im_size[0] / 2))
    pix_y = int((x_y[1] / size_pix) + (im_size[1] / 2))
    pix_x = max(0, min(pix_x, im_size[0] - 1))
    pix_y = max(0, min(pix_y, im_size[1] - 1))
    return pix_x, pix_y

def is_reachable(im, pix):
    """
    Determine if a pixel is reachable, meaning it has at least one neighboring pixel that is free.
    Uses an eight-connected approach to check neighboring pixels.
    """
    for neighbor in path_planning.eight_connected(pix):
        if 0 <= neighbor[0] < im.shape[1] and 0 <= neighbor[1] < im.shape[0]:
            if path_planning.is_free(im, neighbor):
                return True
    return False

def find_all_possible_goals(im, step=60):
    """
    Identify all potential goal locations in a map where each goal is an unseen pixel adjacent to a free pixel.
    Reduces computation by checking pixels in steps, thus skipping some pixels for efficiency.
    """
    possible_goals = []
    height, width = im.shape
    for i in range(1, width - 1, step):  # Skip edges to avoid out-of-bounds
        for j in range(1, height - 1, step):
            if path_planning.is_unseen(im, (i, j)) and is_reachable(im, (i, j)):
                possible_goals.append((i, j))
    return possible_goals

def find_best_point(im, possible_points, robot_loc):
    """
    Select the best exploration point based on proximity to the robot's current location.
    Chooses the point from the list of possible points that is closest to the robot.
    """
    return min(possible_points, key=lambda p: np.linalg.norm(np.array(robot_loc) - np.array(p)))

def find_waypoints(im, path, sampling_interval=20):
    """
    Generates waypoints from a given path by sampling points at specified intervals.
    Ensures the path reaches the goal by adding the last point of the path if not included in the sampled waypoints.
    """
    rospy.loginfo(f"Received path with length: {len(path)}")
    rospy.loginfo(f"Sampling interval: {sampling_interval}")

    if not path:
        rospy.loginfo("Received empty path for waypoint generation")
        return []

    if sampling_interval <= 0:
        rospy.loginfo("Invalid sampling interval, setting to default 5")
        sampling_interval = 5

    try:
        waypoints = [path[i] for i in range(0, len(path), sampling_interval)]
    except IndexError as e:
        rospy.loginfo(f"Error in downsampling path: {e}")
        return []

    if waypoints[-1] != path[-1]:
        waypoints.append(path[-1])

    rospy.loginfo(f"Generated {len(waypoints)} waypoints")
    return waypoints

# Main function for testing
if __name__ == '__main__':
    im, im_thresh = path_planning.open_image("map.pgm")
    robot_start_loc = (1940, 1953)
    all_unseen = find_all_possible_goals(im_thresh)
    best_unseen = find_best_point(im_thresh, all_unseen, robot_loc=robot_start_loc)
    plot_with_explore_points(im_thresh, zoom=0.1, robot_loc=robot_start_loc, explore_points=all_unseen, best_pt=best_unseen)
    path = path_planning.dijkstra(im_thresh, robot_start_loc, best_unseen)
    waypoints = find_waypoints(im_thresh, path)
    path_planning.plot_with_path(im, im_thresh, zoom=0.1, robot_loc=robot_start_loc, goal_loc=best_unseen, path=waypoints)
    print("Done")
