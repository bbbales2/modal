#%%

import numpy
import time
import scipy
import sympy
import os
os.chdir('/home/bbales2/modal')

import rus
reload(rus)
#%%

## Dimensions for TF-2
X = [0.005969, 0.005954, 0.005940]
Y = [0.006928, 0.006944, 0.006928]
Z = [0.009941, 0.009931, 0.009929]

#Sample density
density = [4374.7, 4360.3, 4357.8]#4401.695921 #Ti-64-TF2

c110 = 2.0
anisotropic0 = 2.0
c440 = 1.0
c120 = -(c440 * 2.0 / anisotropic0 - c110)

# Standard deviation around each mode prediction
std0 = 5.0

# Ti-64-TF2 Test Data
data = numpy.array([[141.637,
174.673,
185.173,
236.508,
237.381,
254.555,
260.304,
272.787,
306.617,
314.545,
318.634,
325.971,
332.118,
332.756,
352.026,
368.510,
369.332,
372.104,
385.775,
387.618,
393.065,
394.183,
395.898,
397.460,
405.838,
405.934,
412.396,
421.095,
424.627,
425.232], [142.332,
175.553,
186.464,
236.831,
238.155,
255.504,
261.798,
274.306,
309.218,
314.150,
319.870,
327.065,
331.905,
332.991,
353.229,
368.771,
371.028,
372.809,
386.602,
389.362,
395.510,
396.644,
397.940,
398.579,
407.424,
407.628,
414.514,
423.969,
426.060,
427.477], [142.112,
175.223,
186.017,
236.992,
238.101,
255.777,
262.090,
273.800,
309.064,
314.686,
319.763,
326.922,
332.430,
333.835,
353.427,
369.428,
371.059,
373.194,
387.143,
389.520,
395.632,
396.598,
398.799,
398.983,
408.047,
408.678,
415.174,
423.374,
427.188,
427.463]])[:, :25]

#%%

# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50
# start epsilon at .0001 and try larger values like .0005 after running for a while
# epsilon is timestep, we want to make as large as possibe, wihtout getting too many rejects
epsilon = 0.0001

# Set this to true to debug the L and eps values
debug = False

#%%

reload(rus)

c11, anisotropic, c44 = sympy.symbols('c11 anisotropic c44')

c12 = sympy.sympify("-(c44 * 2.0 / anisotropic - c11)") # The c11 and c44 and anisotropic are the same as above

C = sympy.Matrix([[c11, c12, c12, 0, 0, 0],
                  [c12, c11, c12, 0, 0, 0],
                  [c12, c12, c11, 0, 0, 0],
                  [0, 0, 0, c44, 0, 0],
                  [0, 0, 0, 0, c44, 0],
                  [0, 0, 0, 0, 0, c44]])

hmc = rus.HMC(density = density, X = X, Y = Y, Z = Z,
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : c110, anisotropic : anisotropic0, c44 : c440, 'std' : std0 }, # Parameters
              rotations = [0, 1, 2],
              T = 1.0)

hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)
hmc.sample(steps = 2, debug = True)
#%%
hmc.set_timestepping(epsilon = epsilon * 10.0, L = 50)
hmc.sample(debug = False)#True)#False)#True)
#%%
hmc.derivative_check()
#%%
hmc.set_timestepping(epsilon = epsilon * 20, L = 75)
hmc.sample(debug = True)#False)#True)
#%%
hmc.print_current()
#%%
hmc.posterior_predictive(plot = True, lastN = 200, which_samples = [0, 2])
