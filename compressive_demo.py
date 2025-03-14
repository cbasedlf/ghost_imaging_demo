"""
Code to recover an image from a Ghost Imaging experiment without using the conventional 
approach (correlations / linear combination of speckles).
Three options are shown: using the pseudoinverse method (i.e., minimizing the l2 norm), a
Compressive Sensing approach with a regularization term that minimizes the l1 norm (i.e.
using sparsity assumptions), and a denoising method that minimizes the TV-norm.

@author: F. Soldevila
"""
#%% Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Loading generated speckles 
import h5py

# Libraries for image manipulation (load objects, resize them to speckle size, make animations)
from skimage.transform import resize
from PIL import Image

from scipy.linalg import hadamard
import pylops
import pyproximal

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

meas_num = 256 # Choose number of measurements

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

#%% Recovery using the pseudoinverse (l2 minimization)
obj_pinv = np.zeros((speckles.shape[0] , speckles.shape[1])) # Initialization
Spinv = np.linalg.pinv( S[0:meas_num,:])

obj_pinv_vector = Spinv @ y
obj_pinv = obj_pinv_vector.reshape((speckles.shape[0],speckles.shape[0]))

#%% Solve using FISTA, l1-norm in Hadamard space as regularizer.
# https://pyproximal.readthedocs.io/en/stable/tutorials/twist.html

# Create Hadamard matrix
H = hadamard(speckles.shape[0] * speckles.shape[1])
# Define measurement operator (in sparse space)
A = S[0 : meas_num, :] @ H
# Define operators into pylops variables for solving the problem
Sop = pylops.MatrixMult(S[0:meas_num, :])
Aop = pylops.MatrixMult(A)
Hop = pylops.MatrixMult(H)

# Define objective function to minimize (useful for storing intermediate
# values of the objective function at each iteration of the algo)
def callback(x, pf, pg, eps, cost):
    cost.append(pf(x) + eps * pg(x))
    
# Define algorithm parameters
L = np.abs((Aop.H * Aop).eigs(1)[0])
tau = 0.95 / L
eps = 6e6 # This controls how much relevance you give to l1-minimization
maxit = 500 # Maximum number of iterations for the algo

# FISTA solver
l1 = pyproximal.proximal.L1() # Define l1-norm term
l2 = pyproximal.proximal.L2(Op = Aop, b = y) # Define l2-norm term
costf = [] # Initialize cost function
# Run the optimizer
x_fista = \
    pyproximal.optimization.primal.ProximalGradient(l2, l1, tau = tau,
                    x0 = np.zeros((speckles.shape[0]**2,1)),
                    epsg = eps, niter = maxit, 
                    acceleration = 'fista', show = True,
                    callback = lambda x: callback(x, l2, l1, eps, costf))
niterf = len(costf)
# Reshape solution into 2D array, for visualization
obj_fista_had = np.reshape(Hop.dot(x_fista), 
                              (speckles.shape[0], speckles.shape[1]))

#%% Solve using a denoising approach (l2-norm, TV-norm). Follows the example from:
# https://pyproximal.readthedocs.io/en/stable/tutorials/denoising.html#sphx-glr-tutorials-denoising-py

# Build Gradient operator
sampling = 1.
Gop = pylops.Gradient(dims = (speckles.shape[0], speckles.shape[1]),
                      sampling = sampling, edge = False,
                      kind = 'forward', dtype = 'float64')
Ltv = 8. / sampling ** 2 # maxeig(Gop^H Gop)

# Define terms of the objective function
# L2 data term
l2tv = pyproximal.L2(b = obj_ghost.ravel())
# Isotropic TV  term
sigma = 6e2 #Hyperparameter for TV-term. Determines strength of TV regularization
l1iso = pyproximal.L21(ndim = 2, sigma = sigma) # Define l21-norm term (for isotropic TV)

# Define algorithm parameters
tautv = 1 / np.sqrt(Ltv)
mutv = 1. / (tautv*Ltv)

# Primal-dual solver
x_tv = pyproximal.optimization.primaldual.PrimalDual(l2tv, l1iso, Gop,
                                    tau = tautv, mu = mutv, theta = 1.,
                                    x0 = np.zeros_like(obj_ghost.ravel()),
                                    niter = 100)
# Reshape solution into 2D array, for visualization
obj_TV = np.reshape(x_tv, (speckles.shape[0], speckles.shape[1]))

#%% Show results
fig, axes = plt.subplots(nrows = 1, ncols = 5, figsize = (14,4))
axes[0].imshow(obj)
axes[0].set_title('Ground truth')
axes[1].imshow(obj_ghost)
axes[1].set_title('Classical (correlations)')
axes[2].imshow(obj_pinv)
axes[2].set_title('Pseudoinverse (l2-min.)')
axes[3].imshow(obj_fista_had)
axes[3].set_title('FISTA (l2/l1-min.)')
axes[4].imshow(obj_TV)
axes[4].set_title('TV-norm min.')
