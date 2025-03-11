"""
Ghost imaging demo using predefined speckle patterns

@author: F. Soldevila
"""

#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Loading generated speckles 
import h5py

# Libraries for image manipulation (load objects, resize them to speckle size, make animations)
from skimage.transform import resize
from PIL import Image

#%% Load speckle patterns. You need to generate a file with speckle_generation.py first

speckles_file = 'speckles_64px_65536img' # Filename

# Open file, extract speckles to workspace
with h5py.File(speckles_file + '.h5','r') as f:
    speckles = f['speckles'][:].squeeze()

# Build measurement matrix (each row containes a speckle pattern, reshaped as a row vector)
S = np.reshape(np.moveaxis(speckles,2,0), 
               (speckles.shape[2], speckles.shape[0] * speckles.shape[1]))

#%% Load object. Objects are stored in the folder /objects as .png images

object_name = 'ghost.png' # Object Filename

# Convert to numpy array and resize to speckle size
obj = np.asarray(Image.open('./objects/' + object_name).convert('L'))
obj = resize(obj, (speckles.shape[0],speckles.shape[0])) 

#%% Simulate measurements

meas_num = 512 # Choose number of measurements

# Reshape object into column vector
obj_vec = np.reshape(obj, (speckles.shape[0] * speckles.shape[1], 1))

# Simulate the projection + integration
y = S[0:meas_num, :] @ obj_vec 

#%% Recovery using classical ghost imaging (correlations)

obj_ghost = np.zeros((speckles.shape[0] , speckles.shape[1])) # Initialization
intermediate_ghost = [] # Variable to store intermediate results (used for visualization later)

# Do the recovery: sum of the correlations for successive speckles
for idx in range(meas_num):
    obj_ghost += (y[idx] - np.mean(y)) * speckles[:, :, idx]
    intermediate_ghost.append(np.copy(obj_ghost)) # Store current recovery
obj_ghost /= meas_num # Normalize final recovery
# Reshape into numpy array
obj_ghost_all = np.moveaxis(np.array(intermediate_ghost),0,2)

#%% Create animation showing the recovery
rate = 1/24 # Resfresh rate of the animation

# Figure properties
fig_size = (12,6) # Figure size
fontsize = 10 # Font size

# Create figure
fig = plt.figure(layout = 'constrained', figsize = fig_size)
fig.suptitle('Ghost imaging recovery', fontsize='xx-large') # Set title
subfigs = fig.subfigures(nrows = 2, ncols = 1, wspace = 0.07) # Create subfigures
# Create four subfigs in the first row of the figure
ax_gt, ax_spck, ax_proj, ax_rec = subfigs[0].subplots(nrows = 1, ncols = 4, sharey = True)
im1 = ax_gt.imshow(obj) # Show ground truth object
ax_gt.set_title('Ground truth object', fontsize = fontsize) # Set title
im2 = ax_spck.imshow(speckles[:, :, 0]) # Show speckle pattern
ax_spck.set_title(r'$Speckle_0$', fontsize = fontsize) # Set title
im3 = ax_proj.imshow(obj * speckles[:, :, 0]) # Show multiplication of object and speckle
ax_proj.set_title(r' Object $\times$ Speckle pattern # 0', fontsize = fontsize) # Set title
im4 = ax_rec.imshow(intermediate_ghost[0]) # Show recovery
ax_rec.set_title(r'Recovery: $\sum_{i=0}^{1}{[y_{i} - <y>] \times Speckle_i}$',
                 fontsize = fontsize) # Set title
# Create one subfig in the second row of the figure
ax_int = subfigs[1].subplots(nrows = 1, ncols = 1) 
lineplot = ax_int.plot(np.arange(0, 1, step = 1), y[0:1])[0] # Plot photocurrent
ax_int.set_xlim([0, meas_num]) # Set x-axis limits
ax_int.set_ylim([np.min(y), np.max(y)]) # Set y-axis limits
ax_int.set_xlabel('Speckle #') # Set axis label
ax_int.set_ylabel('Intensity (a.u.)') # Set axis label

# Function to refresh each frame of the animation
def update_plots(idx):
    im2.set_data(speckles[:, :, idx])
    ax_spck.set_title(f'Speckle$_{{{idx}}}$', fontsize = fontsize)
    im3.set_data(obj * speckles[:, :, idx])
    ax_proj.set_title(rf'Object $\times$ Speckle$_{{{idx}}}$', fontsize = fontsize)
    im4.set_data(intermediate_ghost[idx])
    im4.set_clim(vmin = np.min(intermediate_ghost[idx]), vmax = np.max(intermediate_ghost[idx]))
    ax_rec.set_title(r'Recovery: $\sum_{i=0}^' + rf'{{{idx}}}'+ r'{[y_{i} - <y>] \times Speckle_i}$', fontsize = fontsize)
    
    lineplot.set_xdata(np.arange(0, idx, step = 1))
    lineplot.set_ydata(y[0:idx])

# Create animation (comment / uncoment to create it. Quite slow for large meas_num)
anim = animation.FuncAnimation(fig, update_plots, frames = meas_num,
                               interval = rate, repeat = False)
# Save to file
# anim.save('ghost_recovery.mp4', writer = 'ffmpeg', fps =  24, bitrate = 24000, dpi = 300)
