import os
import gc 
import stltovoxel
import numpy as np
from numpy import asarray
from PIL import Image

# Directory Names
directory_stl_files = './stl_files' # Put the stl files in this folder
directory_npy_files = './npy_files' # Name of directory where to save the npy files 
directory_sliced = './sliced'       # A in-between folder where the sliced pictures will be saved

# Create All Paths if they don't exist for further task : important
if not os.path.exists(directory_stl_files): os.mkdir(directory_stl_files)
if not os.path.exists(directory_npy_files): os.mkdir(directory_npy_files)
if not os.path.exists(directory_sliced): os.mkdir(directory_sliced)

# Target depth,height,width
target_d = 70
target_h = 70
target_w = 70

# Generate the sliced images from .stl files and save it in directory_sliced path
allFiles = os.listdir(directory_stl_files)
for fileIndex, filenameWithExtension in enumerate(allFiles):
    print("Slicing: " + filenameWithExtension)
    f = os.path.join(directory_stl_files, filenameWithExtension)
    if os.path.isfile(f) and filenameWithExtension.endswith('.stl'):
        filename = filenameWithExtension.rsplit(".", 1)[0]
        
        # Create Folder if not created
        if not os.path.exists(directory_sliced+'/'+filename): os.mkdir(directory_sliced+'/'+filename)

        # Run STL-TO-VOXEL with resolution=target_d
        stltovoxel.convert_file(directory_stl_files+'/'+filenameWithExtension,directory_sliced+'/'+filename+'/'+'slice.png',resolution=target_d)
        


# Resize the sliced files as per the target_h and target_w
allFolders = os.listdir(directory_sliced)
for fileIndex, folderName in enumerate(allFolders):
    print("Resizing: " + folderName)
    allSliceFiles = os.listdir(directory_sliced+'/'+folderName)
    for fileIndex2, filenameWithExtension in enumerate(allSliceFiles):
        im = Image.open(directory_sliced+'/'+folderName+'/'+filenameWithExtension)
        im_resized = im.resize((target_h,target_w), Image.Resampling.LANCZOS)
        im_resized.save(directory_sliced+'/'+folderName+'/'+filenameWithExtension, "PNG")


# Creating npy files from the resized sliced images of the stl file
allFiles = os.listdir(directory_sliced)
for fileIndex, filenameWithExtension in enumerate(allFiles):
    print("Creating npy: " + filenameWithExtension)
    f = os.path.join(directory_sliced, filenameWithExtension)
    stacked_image = np.zeros((target_d,target_h,target_w), dtype=np.float64)
    allSlicesOfFileIndex = os.listdir(f)
    allSlicesOfFileIndex = sorted(allSlicesOfFileIndex)
    allSlicesOfFileIndex = allSlicesOfFileIndex[0:target_d]
    for idx, slicedPNG in enumerate(allSlicesOfFileIndex):
        image = Image.open(f+'/'+slicedPNG).convert('L')
        arr = asarray(image).astype('float64')
        stacked_image[idx] = arr

    np.save(directory_npy_files+'/'+filenameWithExtension+'.npy', stacked_image)