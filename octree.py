import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
from concurrent.futures import ProcessPoolExecutor

class OctreeNode:
    def __init__(self, boundary: Tuple[float, float, float, float, float, float], max_points: int = 100, depth: int = 0):
        self.boundary = boundary
        self.max_points = max_points
        self.depth = depth
        self.children: List[OctreeNode] = []
        self.points: np.ndarray = np.array([])
        self.mean_point: np.ndarray = np.array([])
        self.sphere_center = self._calculate_sphere_center()
        self.sphere_radius = self._calculate_sphere_radius()
        self.optimize_sphere_position()

    def _calculate_sphere_center(self) -> np.ndarray:
        return np.array([(self.boundary[i] + self.boundary[i+3]) / 2 for i in range(3)])

    def _calculate_sphere_radius(self) -> float:
        return max((self.boundary[i+3] - self.boundary[i]) / 2 for i in range(3))

    def optimize_sphere_position(self):
        if self.points.size == 0:
            return

        # Calculate the mean position of points
        mean_position = np.mean(self.points, axis=0)

        # Calculate the maximum allowed displacement
        max_displacement = np.array([
            min(mean_position[i] - self.boundary[i], self.boundary[i+3] - mean_position[i])
            for i in range(3)
        ])

        # Limit the displacement to ensure the sphere stays within the node boundaries
        displacement = np.clip(mean_position - self.sphere_center, -max_displacement, max_displacement)

        # Update the sphere center
        self.sphere_center += displacement * 0.8  # Use 80% of the calculated displacement for safety

    def insert(self, points: np.ndarray) -> None:
        points_in_sphere = self._filter_points_in_boundary(points)
        
        if len(points_in_sphere) == 0:
            return
        
        if len(self.points) + len(points_in_sphere) <= self.max_points or self.depth >= 8:
            self.points = np.vstack([self.points, points_in_sphere]) if self.points.size else points_in_sphere
            self.mean_point = np.mean(self.points, axis=0)
            self.optimize_sphere_position()
        else:
            if not self.children:
                self._subdivide()
            
            for child in self.children:
                child.insert(points_in_sphere)
            
            self.points = np.array([])
            if points_in_sphere.size > 0:
                self.mean_point = np.mean(points_in_sphere, axis=0)
                self.optimize_sphere_position()  # Optimize even for non-leaf nodes

    def _subdivide(self) -> None:
        mid_x, mid_y, mid_z = self.sphere_center

        octants = [
            (self.boundary[0], self.boundary[1], self.boundary[2], mid_x, mid_y, mid_z),
            (mid_x, self.boundary[1], self.boundary[2], self.boundary[3], mid_y, mid_z),
            (self.boundary[0], mid_y, self.boundary[2], mid_x, self.boundary[4], mid_z),
            (mid_x, mid_y, self.boundary[2], self.boundary[3], self.boundary[4], mid_z),
            (self.boundary[0], self.boundary[1], mid_z, mid_x, mid_y, self.boundary[5]),
            (mid_x, self.boundary[1], mid_z, self.boundary[3], mid_y, self.boundary[5]),
            (self.boundary[0], mid_y, mid_z, mid_x, self.boundary[4], self.boundary[5]),
            (mid_x, mid_y, mid_z, self.boundary[3], self.boundary[4], self.boundary[5])
        ]

        self.children = [OctreeNode(octant, self.max_points, self.depth + 1) for octant in octants]

        for child in self.children:
            child_points = child._filter_points_in_boundary(self.points)
            if len(child_points) > 0:
                child.insert(child_points)

        self.points = np.array([])

    def _filter_points_in_boundary(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points
        # First, filter points within the cube
        cube_mask = np.all((self.boundary[:3] <= points) & (points < self.boundary[3:]), axis=1)
        points_in_cube = points[cube_mask]
        
        # Then, filter points within the sphere using the optimized center
        distances = np.linalg.norm(points_in_cube - self.sphere_center, axis=1)
        sphere_mask = distances <= self.sphere_radius
        return points_in_cube[sphere_mask]

    def get_points_for_lod(self, camera_position: np.ndarray, lod_factor: float) -> np.ndarray:
        distance = np.linalg.norm(camera_position - self.sphere_center)
        if distance > self.sphere_radius * lod_factor or not self.children:
            if self.points.size:
                return self.points
            elif self.mean_point.size and np.all(np.isfinite(self.mean_point)):
                return self.mean_point.reshape(1, -1)
            else:
                return np.array([])
        else:
            child_points = [child.get_points_for_lod(camera_position, lod_factor) for child in self.children]
            return np.vstack([points for points in child_points if points.size > 0])

class Octree:
    def __init__(self, boundary: Tuple[float, float, float, float, float, float], max_points: int = 100):
        self.root = OctreeNode(boundary, max_points)

    def insert(self, points: np.ndarray) -> None:
        self.root.insert(points)

    def get_points_for_lod(self, camera_position: np.ndarray, lod_factor: float = 2.0) -> np.ndarray:
        return self.root.get_points_for_lod(camera_position, lod_factor)

    def visualize(self, ax: Optional[Axes3D] = None, max_points: int = 1000000, camera_position: np.ndarray = None, lod_factor: float = 1.5) -> None:
        if ax is None:
            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

        # Always use get_all_points to ensure we have points to display
        points = self.get_all_points()

        if points.size:
            # Filter out invalid points
            valid_mask = np.all(np.isfinite(points), axis=1)
            points = points[valid_mask]

            if len(points) > max_points:
                indices = np.random.choice(len(points), max_points, replace=False)
                points = points[indices]

            if len(points) > 0:
                min_height, max_height = points[:, 2].min(), points[:, 2].max()
                if np.isclose(min_height, max_height):
                    normalized_heights = np.zeros(len(points))
                else:
                    normalized_heights = (points[:, 2] - min_height) / (max_height - min_height)

                # Create a custom colormap with green, orange, and red
                cmap = colors.LinearSegmentedColormap.from_list("custom", ["green", "orange", "red"])
                
                scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                                     c=normalized_heights, cmap=cmap, s=1, alpha=0.8)
                
                plt.colorbar(scatter, ax=ax, label='Elevation')

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.set_title('Octree Point Cloud Visualization')

                # Set equal aspect ratio safely
                x_range = np.ptp(points[:, 0])
                y_range = np.ptp(points[:, 1])
                z_range = np.ptp(points[:, 2])
                max_range = max(x_range, y_range, z_range)
                if max_range > 0:
                    ax.set_box_aspect((x_range/max_range, y_range/max_range, z_range/max_range))

                plt.tight_layout()
                plt.show()
            else:
                print("No valid points to visualize after filtering.")
        else:
            print("No points to visualize. The Octree might be empty.")

    def _visualize_spheres(self, ax: Axes3D):
        def draw_sphere(node):
            if node.points.size or node.children:
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = node.sphere_center[0] + node.sphere_radius * np.cos(u) * np.sin(v)
                y = node.sphere_center[1] + node.sphere_radius * np.sin(u) * np.sin(v)
                z = node.sphere_center[2] + node.sphere_radius * np.cos(v)
                ax.plot_wireframe(x, y, z, color="b", alpha=0.1)

            for child in node.children:
                draw_sphere(child)

        draw_sphere(self.root)

    def get_all_points(self):
        def collect_points(node):
            if node.points.size:
                return node.points
            child_points = [collect_points(child) for child in node.children if child.children or child.points.size]
            return np.vstack(child_points) if child_points else np.array([])
        
        return collect_points(self.root)

def parallel_octree_insert(octree: Octree, points: np.ndarray, num_processes: int = 4) -> None:
    chunk_size = len(points) // num_processes
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [executor.submit(octree.insert, points[i:i+chunk_size]) for i in range(0, len(points), chunk_size)]
        for future in futures:
            future.result()
