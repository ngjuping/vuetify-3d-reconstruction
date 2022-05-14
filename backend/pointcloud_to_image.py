import numpy as np
import open3d as o3d
import pptk
import math

# Read .ply file
input_files = ["chair.ply","waiting_room.ply","plant1.ply"]
v = None
for input_file in input_files:
    pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud

    # Convert open3d format to numpy array
    # Here, you have the point cloud in numpy format. 
    point_cloud_in_numpy = np.asarray(pcd.points)
    scalars = np.arange(point_cloud_in_numpy.shape[0])
    v = pptk.viewer(point_cloud_in_numpy, scalars)
    v.set(point_size=0.0001)
    v.set(r=5)
    v.set(phi=math.pi/2)
    v.set(theta=-math.pi/2)
    v.set(show_grid=False)
    v.set(show_axis=False)
    v.set(show_info=False)
    v.set(bg_color=(0,0,0,1))
    v.color_map([[0, 0, 0], [1, 1, 1]])
    v.capture(f'{input_file}.png')

