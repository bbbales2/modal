#%%

import numpy
import time
import scipy
import sympy
import os
import rus

## Dimensions for TF-2
Xs = [0.011959, 0.012, 0.01198]
Ys = [0.013953, 0.01399, 0.01397]
Zs = [0.019976, 0.01999, 0.020]

#Sample density
densities = [8701.0, 0.0290599 / (Xs[1] * Ys[1] * Zs[1]), 0.0290864 / (Xs[2] * Ys[2] * Zs[2])]#4401.695921 #Ti-64-TF2

c110 = 2.0
anisotropic0 = 2.00
c440 = 1.0
c120 = -(c440 * 2.0 / anisotropic0 - c110)

# Standard deviation around each mode prediction
std0 = 5.0

data1 = numpy.array([71.25925, 75.75875, 86.478, 89.947375, 111.150125,
                     112.164125, 120.172125, 127.810375, 128.6755, 130.739875,
                     141.70025, 144.50375, 149.40075, 154.35075, 156.782125,
                     157.554625, 161.0875, 165.10325, 169.7615, 173.44925,
                     174.11675, 174.90625, 181.11975, 182.4585, 183.98625,
                     192.68125, 193.43575, 198.793625, 201.901625, 205.01475])

data2 = numpy.array([71.097, 75.589, 86.227, 89.863, 110.726,
                     111.696, 119.9796667, 127.452, 128.304, 130.481,
                     141.407, 143.84, 149.066, 153.761, 156.37,
                     156.979, 160.307, 164.661, 168.963, 172.473,
                     173.421, 174.196, 180.633, 181.54, 183.136,
                     192.122, 193.141, 198.24, 201.402, 204.214])

data3 = numpy.array([71.111, 75.578, 86.207, 89.866, 110.734,
                     111.728, 120.024, 127.47, 128.312, 130.463,
                     141.437, 143.897, 149.073, 153.828, 156.404,
                     157.027, 160.377, 164.709, 169.081, 172.609,
                     173.449, 174.235, 180.579, 181.674, 183.27,
                     192.167, 193.14, 198.27, 201.434, 204.257])

data = numpy.array([data1, data2, data3])

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

hmc = rus.HMC(density = densities, X = Xs, Y = Ys, Z = Zs,
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : c110, anisotropic : anisotropic0, c44 : c440, 'std' : std0 }, # Parameters
              rotations = [0, 1, 2],
              T = 1.0,
              stdMin = 0.0,
              tol = 1e-3)

hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)
hmc.sample(steps = 10, debug = False, silent = True)
hmc.set_timestepping(epsilon = epsilon * 2.0, L = 50, param_scaling = { 'std' : 2.0 })

while True:
    hmc.sample(steps = 1, debug = False, silent = True)#False)#True)
    print ", ".join("{0}, {1}".format(a, b[0]) for a, b in zip(*hmc.format_samples(1)))
