"""
Code to generate random speckle images. Useful for simulations of ghost imaging

The procedure follows the following steps:
1 - Create a random phase mask with a set of spatial properties (~grain size & ~grain height). 
This phase mask simulates a rough surface (with a given grain size and height)
2 - Place a circular pupil with a given radius
3 - Propagate that phase distribution to far-field via FFT, generating speckle. You can use some
padding to create better-resolution speckles

Changing the phase mask properties will create different speckles characteristics (for example, 
big phase jumps will create speckles that cover larger FoVs). You can play with:
        - Pupil size (radius)
        - Random mask ~grain size (lateral) and height (variables corr_width and scatter_strength)

You can store the generated speckles at the end of the code for later use

I use my optsim library (https://github.com/cbasedlf/optsim) for speckle generation, 
FFT implementation, and some visualization. I included that in the package so there is no need
to download/install it.
The rest are common python libraries (NumPy, h5Py)


@author: F. Soldevila
"""
#%% Import libraries
import numpy as np
import optsim as ops # Speckle generation
import h5py # Saving generated speckles 

#%% Define physical parameters of the simulation

wvl = 532e-9 # Wavelength, in m
aperture_size = 100e-6 # Pphysical size of the aperture, in m
pxnum = 32 # Total number of pixels for the aperture (before padding)
pxsize = aperture_size / pxnum # Pixel size, in m

#%% Define number of speckles to create
specklenum = 2**8

#%% Generate thin scattering media to create speckles. 
# You can tweak [corr_width] and  [scatter_strength] to play with how the speckles look

# Correlation width of the rough surface that will generate the spekles, in m
corr_width = 4e-6
# Strength of the scatterer (number of 2*pi phase jumps the surface will introduce)
scatter_strength = 16

scat = [] # Initialization
for idx in range(specklenum):
    scat.append(ops.thin_scatter(size = aperture_size, pxnum = pxnum, 
                        corr_width = corr_width, strength = scatter_strength) )

#%% Mask the scattering media (place a circular pupil)
mask_radius = 0.8 # Pupil size (ratio of total aperture)
# Create mask
mask, _, _ = ops.circAp(totalSize = pxnum, radius = mask_radius) 

for idx in range(specklenum):
    scat[idx].thin_scat *= mask

#%% Propagate a field through the scattering medium to generate speckles.
padsize = 16 # Padding size for the FFT process. End image size will be [2 * padsize + pxsize]
#Far field propagation (Fourier transform):
speckles = [] # Initialization
for idx in range(specklenum):
    speckles.append( np.abs( ops.ft2(np.pad(scat[idx].thin_scat,
                                   [padsize, padsize], mode = 'constant'), 1) )**2 )
# Convert to numpy array (and reorder axes)
speckles = np.array(speckles)
speckles = np.moveaxis(speckles, 0, 2)
# Convert speckles to uint8 (to reduce storage needs)
speckles = speckles / np.max(speckles)
speckles *= 255
speckles = speckles.astype(np.uint8)

# #Show video of speckles (comment/uncomment if you want to take a look at the speckles)
# ops.show_vid(speckles, rate = 200)

#%% Store speckles into h5 file
# filename = 'speckles_64px_65536img'

# with h5py.File(filename + '.h5', 'w') as f:
#     # Save experimental data
#     f.create_dataset('speckles', data = speckles)