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
Y = 0.01079#0.009057
Z = 0.013025#0.013199

#Sample density
density = 8114.300888#4401.695921 #Ti-64-TF2

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
341.585])
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
                     [0, 0, 0, 0, 0, (c11 - c12) / 2.0]])

hmc = rus.HMC(tol = 1e-3,
              density = density, X = X, Y = Y, Z = Z,
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : 2.0, c12 : 1.0, c13 : 1.0, c33 : 1.0, c44 : 1.0, 'std' : 5.0 }, # Parameters
              constrained_positive = [c11, c12, c13, c33, c44],
              rotations = True,
              T = 1.0)

hmc.set_labels({ c11 : 'c11', c12 : 'c12', c13 : 'c13', c33 : 'c33', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)
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
hmc.posterior_predictive(plot = True, lastN = 200)
plt.title('Posterior predictive', fontsize = 72)
plt.xlabel('Mode', fontsize = 48)
plt.ylabel('Computed - Measured (khz)', fontsize = 48)
plt.tick_params(axis='y', which='major', labelsize=48)
plt.tick_params(axis='x', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches((24, 16))
plt.savefig('dec2/andrewhexagonal/posteriorpredictive.png', dpi = 144)
plt.show()
#%%
#%%
import pickle

f = open('/home/bbales2/modal/dec2/andrew_hexagonal.pkl', 'w')
pickle.dump(hmc, f)
f.close()
