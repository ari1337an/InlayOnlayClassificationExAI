import numpy as np
from stl import mesh
import scipy.ndimage
import os

## create a directory named "data_npy" in the same directory as this file if not already created.
if not os.path.exists("data_npy"):
    os.mkdir("data_npy")
    

## Traverse through the directory and get the path of all the .stl files
for root, dirs, files in os.walk("./Test_files_inlay_onlay/"):
    for file in files:

        # Load the STL file
        mesh_stl = mesh.Mesh.from_file(os.path.join(root, file))

        # Extract the vertices and faces from the mesh
        vertices = mesh_stl.vectors.reshape((-1, 3))
        faces = np.arange(len(vertices)).reshape((-1, 3))

        # Calculate the size of the bounding box that contains the mesh
        xmin, ymin, zmin = np.min(vertices, axis=0)
        xmax, ymax, zmax = np.max(vertices, axis=0)
        xsize, ysize, zsize = int(xmax-xmin)+1, int(ymax-ymin)+1, int(zmax-zmin)+1


        # Create a blank 3D numpy array with dimensions equal to the size of the bounding box
        data = np.zeros((xsize, ysize, zsize))

        # Set the voxel values for the object using the vertices and faces
        # print(data.shape)
        x, y, z = np.indices(data.shape)
        # print("indices")
        # print(x, y, z)
        x += int(xmin)
        y += int(ymin)
        z += int(zmin)
        for triangle in faces:
            v1, v2, v3 = vertices[triangle]
            normal = np.cross(v2-v1, v3-v1)
            d = np.dot(normal, v1)
            data[(normal[0]*x + normal[1]*y + normal[2]*z) >= d] = 1

        # Smooth the voxel values using a Gaussian filter
        data_smoothed = scipy.ndimage.gaussian_filter(data, sigma=1)

        # Resample the voxel array to a desired resolution
        target_shape = (128, 128, 128)
        zoom_factor = np.array(target_shape) / np.array(data_smoothed.shape)
        data_resampled = scipy.ndimage.zoom(data_smoothed, zoom_factor, order=1)

        # Normalize the voxel values using z-score normalization
        mean = np.mean(data_resampled)
        std = np.std(data_resampled)
        data_normalized = (data_resampled - mean) / std

        ## delete .stl from the file name and add .npy
        file = file[:-4] + ".npy"
        
        ## Save the file in the "data_npy" folder
        np.save(os.path.join("data_npy", file), data_normalized)

        print("File saved: ", file)