#%%

import numpy
import time
import scipy
import sympy
import os
import rus

X = 0.007753
Y = 0.009057
Z = 0.013199

#Sample density
density = 4401.695921 #Ti-64-TF2

c110 = 2.0
anisotropic0 = 1.0
c440 = 1.0
c120 = -(c440 * 2.0 / anisotropic0 - c110)

# Standard deviation around each mode prediction
std0 = 5.0

# Ti-64-TF2 Test Data
data = numpy.array([109.076,
136.503,
144.899,
184.926,
188.476,
195.562,
199.246,
208.460,
231.220,
232.630,
239.057,
241.684,
242.159,
249.891,
266.285,
272.672,
285.217,
285.670,
288.796,
296.976,
301.101,
303.024,
305.115,
305.827,
306.939,
310.428,
318.000,
319.457,
322.249,
323.464,
324.702,
334.687,
340.427,
344.087,
363.798,
364.862,
371.704,
373.248])[:30]

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

hmc = rus.HMC(density = [density], X = [X], Y = [Y], Z = [Z],
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : c110, anisotropic : anisotropic0, c44 : c440, 'std' : std0 }, # Parameters
              rotations = False,
              T = 1.0,
              stdMin = 0.3)

hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })

hmc.set_timestepping(epsilon = epsilon, L = 50)
hmc.sample(steps = 10, debug = False, silent = True)
hmc.set_timestepping(epsilon = epsilon * 4.0, L = 100, param_scaling = { 'std' : 40.0 })

while True:
    hmc.sample(steps = 1, debug = False, silent = True)#False)#True)
    print ", ".join("{0}, {1}".format(a, b[0]) for a, b in zip(*hmc.format_samples(1)))
