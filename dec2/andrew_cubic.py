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
#656 samples
#%%
## Dimensions for TF-2
X = 0.0055#0.007753
Y = 0.010785#0.009057
Z = 0.01302#0.013199

#Sample density
density = 8054.10909#4401.695921 #Ti-64-TF2

c110 = 2.0
anisotropic0 = 1.0
c440 = 1.0
c120 = -(c440 * 2.0 / anisotropic0 - c110)

# Standard deviation around each mode prediction
std0 = 5.0

data = numpy.array([86.916,
108.277,
146.582,
155.722,
166.274,
167.510,
173.016,
181.229,
195.209,
204.479,
218.063,
225.679,
240.271,
251.107,
254.079,
258.857,
264.639,
281.978,
286.834,
287.739,
311.693,
318.010,
327.643,
328.771,
336.807,
338.585,
342.069,
344.734,
346.879,
350.192])

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

hmc = rus.HMC(tol = 1e-3,
              density = density, X = X, Y = Y, Z = Z,
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : c110, anisotropic : anisotropic0, c44 : c440, 'std' : std0 }, # Parameters
              rotations = False,
              T = 1.0)

hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)
#hmc.print_current()
#hmc.computeResolutions(1e-3)
hmc.sample(steps = 5, debug = True)
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
import pickle

f = open('/home/bbales2/modal/dec2/andrew_cubic.pkl', 'w')
pickle.dump(hmc, f)
f.close()
#%%
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(*hmc.format_samples()):
    plt.plot(data1)
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 24)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.show()
#%%
for name, data1 in zip(*hmc.format_samples()):
    data1 = data1[-2000:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()