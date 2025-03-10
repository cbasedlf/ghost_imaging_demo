"""
Ghost imaging demo using predefined speckle patterns

@author: F. Soldevila
"""

#%% Import libraries
import numpy as np
import optsim as ops

# Loading generated speckles 
import h5py

# Libraries for image manipulation (load objects, resize them to speckle size, make animations)
from skimage.transform import resize
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

meas_num = 8192 # Choose number of measurements

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
rate = 0.1

fig_size = (8,8)
fontsize = 10

fig = plt.figure(layout = 'constrained', figsize = fig_size)
fig.suptitle('Ghost imaging recovery', fontsize='xx-large')
subfigs = fig.subfigures(nrows = 2, ncols = 1, wspace = 0.07)

ax_gt, ax_spck, ax_proj, ax_rec = subfigs[0].subplots(nrows = 1, ncols = 4, sharey = True)
im1 = ax_gt.imshow(obj)
ax_gt.set_title('Ground truth object', fontsize = fontsize)
im2 = ax_spck.imshow(speckles[:, :, 0])
ax_spck.set_title(r'$Speckle_0$', fontsize = fontsize)
im3 = ax_proj.imshow(obj * speckles[:, :, 0])
ax_proj.set_title(r' Object $\times$ Speckle pattern # 0', fontsize = fontsize)
im4 = ax_rec.imshow(intermediate_ghost[0])
ax_rec.set_title(r'Recovery: $\sum_{i=0}^{1}{[y_{i} - <y>] \times Speckle_i}$', fontsize = fontsize)

ax_int = subfigs[1].subplots(nrows = 1, ncols = 1)
lineplot = ax_int.plot(np.arange(0, 1, step = 1), y[0:1])[0]
ax_int.set_xlim([0, meas_num])
ax_int.set_xlabel('Speckle #')
ax_int.set_ylabel('Intensity (a.u.)')

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
    
    plt.draw()
    plt.show()
    

anim = animation.FuncAnimation(fig, update_plots, frames = meas_num,
                               interval = rate, repeat = False)
