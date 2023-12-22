#!/usr/bin/env python3

import numpy as np
import heapq
import imageio
import rospy

# -------------- Functions for Visualizing Map and Path ---------------
def plot_with_path(im, im_threshhold, zoom=1.0, robot_loc=None, goal_loc=None, path=None):
    """
    Display the map with the robot's location, goal location, and proposed path.
    - im: Original map image.
    - im_threshhold: Thresholded map image.
    - zoom: Zoom level for viewing the map.
    - robot_loc: Robot's location in pixel coordinates.
    - goal_loc: Goal's location in pixel coordinates.
    - path: Proposed path in pixel coordinates.
    """
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(im, origin='lower', cmap="gist_gray")
    axs[0].set_title("Original Image")
    axs[1].imshow(im_threshhold, origin='lower', cmap="gist_gray")
    axs[1].set_title("Threshold Image")

    # Plotting robot location, goal location, and path on the map
    for i in range(2):
        if robot_loc is not None:
            axs[i].plot(robot_loc[0], robot_loc[1], '+r', markersize=10)
        if goal_loc is not None:
            axs[i].plot(goal_loc[0], goal_loc[1], '*g', markersize=10)
        if path is not None:
            for p, q in zip(path[:-1], path[1:]):
                axs[i].plot([p[0], q[0]], [p[1], q[1]], '-y', markersize=2)
                axs[i].plot(p[0], p[1], '.y', markersize=2)
        axs[i].axis('equal')
    
    # Set zoom level for the axes
    width, height = im.shape[1], im.shape[0]
    for ax in axs:
        ax.set_xlim(width / 2 - zoom * width / 2, width / 2 + zoom * width / 2)
        ax.set_ylim(height / 2 - zoom * height / 2, height / 2 + zoom * height / 2)

# -------------- Thresholded Image Functions ---------------
def is_wall(im, pix):
    """
    Determine if a pixel is a wall.
    - im: The image (map).
    - pix: The pixel coordinates (i, j).
    Returns True if the pixel is a wall, False otherwise.
    """
    return im[pix[1], pix[0]] > 0

def is_unseen(im, pix):
    """
    Check if a pixel is unseen (not yet explored).
    - im: The image (map).
    - pix: The pixel coordinates (i, j).
    Returns True if the pixel is unseen, False otherwise.
    """
    return im[pix[1], pix[0]] < 0

def is_free(im, pix):
    """
    Check if a pixel is free space (not a wall or unseen).
    - im: The image (map).
    - pix: The pixel coordinates (i, j).
    Returns True if the pixel is free space, False otherwise.
    """
    return im[pix[1], pix[0]] <= 0

def convert_image(im, wall_threshold, free_threshold):
    """
    Convert an image to a thresholded image marking walls, free space, and unseen areas.
    - im: The input image.
    - wall_threshold: Threshold to indicate a wall.
    - free_threshold: Threshold to indicate free space.
    Returns an image with 0 (free), 255 (wall), and 128 (unseen).
    """
    im_ret = np.zeros((im.shape[0], im.shape[1]), dtype='uint8') + 128
    im_avg = np.mean(im, axis=2) if len(im.shape) == 3 else im
    im_avg = im_avg / np.max(im_avg)
    im_ret[im_avg < wall_threshold] = 0
    im_ret[im_avg > free_threshold] = 255
    return im_ret

# -------------- Neighbor Functions ---------------
def four_connected(pix):
    """
    Generator function for 4-connected neighbors.
    - pix: The i, j location to iterate around.
    Yields neighboring pixels in 4 directions.
    """
    for i in [-1, 1]:
        yield pix[0] + i, pix[1]
    for i in [-1, 1]:
        yield pix[0], pix[1] + i

def eight_connected(pix):
    """
    Generator function for 8-connected neighbors.
    - pix: The i, j location to iterate around.
    Yields neighboring pixels in 8 directions.
    """
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 or j != 0:
                yield pix[0] + i, pix[1] + j

# -------------- Dijkstra's Algorithm ---------------
def dijkstra(im, robot_loc, goal_loc):
    """
    Implement Dijkstra's (or A*) algorithm for pathfinding.
    - im: The thresholded map image.
    - robot_loc: Robot's starting location.
    - goal_loc: Destination location.
    Returns the calculated path.
    """
    rospy.loginfo("Starting Dijkstra's algorithm")
    if not is_free(im, robot_loc):
        rospy.loginfo(f"Start location {robot_loc} is not in the free space of the map")
        return []

    if not is_free(im, goal_loc):
        rospy.loginfo(f"Goal location {goal_loc} is not in the free space of the map")
        return []

    priority_queue = []
    heapq.heappush(priority_queue, (0, robot_loc))
    visited = {robot_loc: (0, None, False)}

    while priority_queue:
        current_dist, current_pix = heapq.heappop(priority_queue)

        if current_pix == goal_loc:
            rospy.loginfo("Goal reached")
            return reconstruct_path(visited, goal_loc, robot_loc)

        if visited[current_pix][2]:  # Skip if node is already visited
            continue

        visited[current_pix] = (visited[current_pix][0], visited[current_pix][1], True)

        for neighbor in eight_connected(current_pix):
            if is_valid_neighbor(im, neighbor):
                g_cost = current_dist + 1
                f_cost = g_cost + heuristic(neighbor, goal_loc)
                if neighbor not in visited or g_cost < visited[neighbor][0]:
                    visited[neighbor] = (g_cost, current_pix, False)
                    heapq.heappush(priority_queue, (f_cost, neighbor))

    rospy.loginfo("Path not found")
    return []

def is_valid_neighbor(im, neighbor):
    """
    Check if a neighbor is valid (within bounds and free).
    - im: The map image.
    - neighbor: Neighbor pixel coordinates.
    Returns True if the neighbor is valid, False otherwise.
    """
    return 0 <= neighbor[0] < im.shape[1] and 0 <= neighbor[1] < im.shape[0] and is_free(im, neighbor)

def reconstruct_path(visited, goal, start):
    """
    Reconstruct the path from the visited nodes.
    - visited: Dictionary of visited nodes with their parent information.
    - goal: Goal location.
    - start: Starting location.
    Returns the reconstructed path.
    """
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = visited[current][1]
    path.append(start)
    path.reverse()
    rospy.loginfo("Path reconstructed")
    return path

# -------------- Heuristic Function ---------------
def heuristic(current, goal):
    """
    Heuristic function for A* algorithm, using Euclidean distance.
    - current: Current node coordinates.
    - goal: Goal node coordinates.
    Returns the heuristic value (distance).
    """
    return np.sqrt((current[0] - goal[0])**2 + (current[1] - goal[1])**2)

# -------------- Image Opening Function ---------------
def open_image(im_name):
    """
    Open an image and its associated YAML file to threshold it.
    - im_name: Name of the image in the Data directory.
    Returns the original and thresholded images.
    """
    from os import open
    im = imageio.imread("Data/" + im_name)
    wall_threshold = 0.7
    free_threshold = 0.9

    try:
        yaml_name = "Data/" + im_name[:-3] + "yaml"
        with open(yaml_name, "r") as f:
            dict = yaml.load_all(f)
            wall_threshold = dict["occupied_thresh"]
            free_threshold = dict["free_thresh"]
    except:
        pass

    im_thresh = convert_image(im, wall_threshold, free_threshold)
    return im, im_thresh

# -------------- Main Execution ---------------
if __name__ == '__main__':
    import yaml

    # Example usage with a specific map and robot locations
    im, im_thresh = open_image("map.pgm")
    robot_start_loc = (1940, 1953)
    robot_goal_loc = (2135, 2045)
    zoom = 0.1

    path = dijkstra(im_thresh, robot_start_loc, robot_goal_loc)
    plot_with_path(im, im_thresh, zoom=zoom, robot_loc=robot_start_loc, goal_loc=robot_goal_loc, path=path)
    print("Done")
