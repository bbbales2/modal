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

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

##
X = 0.00550
Y = 0.01079
Z = 0.013025

mass = 6.2721e-3

#Sample density
density = mass / (X * Y * Z)

# Ti-64-TF2 Test Data
data = numpy.array([84.824,
104.067,
121.173,
131.982,
146.605,
159.923,
170.609,
177.603,
179.117,
192.316,
213.764,
214.697,
219.536,
220.501,
236.464,
256.867,
262.779,
267.559,
270.429,
280.313,
303.722,
309.733,
313.656,
319.363,
326.227,
331.307,
334.499,
337.853,
338.842,
341.585,
348.237,
350.3363333,
355.5488,
358.4326667,
361.991,
366.2608333,
370.2301667,
371.6196667,
376.2252,
377.9055,
378.729,
380.951,
381.6706667,
386.8276667,
392.6591667,
394.23,
396.2695,
399.8153333,
404.451,
407.6056667])

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

c11, c12, c13, c33, c44 = sympy.symbols('c11 c12 c13 c33 c44')

C = sympy.Matrix([[c11, c12, c13, 0, 0, 0],
                  [c12, c11, c13, 0, 0, 0],
                  [c13, c13, c33, 0, 0, 0],
                  [0, 0, 0, c44, 0, 0],
                  [0, 0, 0, 0, c44, 0],
                  [0, 0, 0, 0, 0, (c11 - c12) / 2.0]], evaluate = False)

hmc = rus.HMC(N = N, # Order of Rayleigh-Ritz approximation
              density = density * 1e-3, X = X * 10, Y = Y * 10, Z = Z * 10,
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : 2.0e-1, c12 : 1.0e-1, c13 : 1.0e-1, c33 : 1.0e-1, c44 : 1.0e-1, 'std' : 5.0 }, # Parameters
              constrained_positive = [c11, c12, c13, c33, c44, 'std']) # Constrain these variables to be positive

hmc.set_labels({ c11 : 'c11', c12 : 'c12', c13 : 'c13', c33 : 'c44', c44 : 'c44', 'std' : 'std' })

hmc.set_timestepping(epsilon = epsilon, L = 50, param_scaling = { 'std' : 1.1 }) # Param scaling -- this means make param std move around about twice as fast as the other parameters

hmc.sample(debug = True)
#%%
hmc.derivative_check()
#%%
reload(rus)

hmc.set_timestepping(epsilon = epsilon * 10, L = 50, param_scaling = { 'std' : 1.1 })

hmc.sample(debug = False)

#%%
reload(rus)

print hmc.saves()
