import matplotlib.pyplot as plt
import numpy as np

# Configurations
directory_stl_files = './stl_files' # Put the stl files in this folder
directory_npy_files = './npy_files' # Name of directory where to save the npy files 
directory_sliced = './sliced'       # A in-between folder where the sliced pictures will be saved
file_to_visualize = 'T1.1.npy'      # File to visualize

# Load the npy file file_to_visualize content in the directory_npy_files folder
images = np.load(directory_npy_files+'/'+file_to_visualize)

# Print the shape of the iamges (Channel x Height x Width)
print("Size of npy file content: ", images.shape)

# Create a grid of subplots
fig, axs = plt.subplots(10, 7, figsize=(14, 20))

# Iterate through each image and plot it in a corresponding subplot
for i, ax in enumerate(axs.flat):
    # Get the current image
    img = images[i]
    
    # Plot the image
    ax.imshow(img, cmap='gray')
    ax.axis('off')  # Turn off the axes
    
# Adjust the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
