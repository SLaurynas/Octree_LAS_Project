import pylas
import os
import numpy as np
from octree import Octree
from multiprocessing import freeze_support

def main():
    las_file_path = os.path.join(os.getcwd(), '2743_1234.las')

    print("Loading LAS file...")
    las = pylas.read(las_file_path)
    print(f"Loaded {len(las.points)} points")

    # Get the bounding box directly from the LAS file
    min_x, min_y, min_z = las.header.mins
    max_x, max_y, max_z = las.header.maxs
    boundary = (min_x, min_y, min_z, max_x, max_y, max_z)
    print(f"Point cloud bounds: {boundary}")

    # Sample points for Octree creation
    max_points = 1000000
    if len(las.points) > max_points:
        print(f"Sampling {max_points} points for Octree creation")
        indices = np.random.choice(len(las.points), max_points, replace=False)
        points = np.vstack((las.x[indices], las.y[indices], las.z[indices])).transpose()
    else:
        points = np.vstack((las.x, las.y, las.z)).transpose()

    # Initialize the root Octree node with the point cloud
    print("Creating modified Octree with embedded spheres...")
    octree = Octree(boundary, max_points=1000)  # Kept max_points per node at 1000
    octree.insert(points)
    print("Octree creation completed.")

    # Visualize points inside Octree spheres
    print("Visualizing points inside Octree spheres...")
    camera_position = np.array([np.mean(boundary[:3]), np.mean(boundary[3:]), max(boundary[5], boundary[2]) * 1.5])
    octree.visualize(max_points=1000000, camera_position=camera_position)
    print("Visualization completed.")

if __name__ == '__main__':
    freeze_support()
    main()
