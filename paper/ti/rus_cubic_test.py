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
hmc.derivative_check()

hmc.set_timestepping(epsilon = epsilon, L = 50)
#hmc.print_current()
#hmc.computeResolutions(1e-3)
hmc.sample(steps = 5, debug = True)
#%%
hmc.set_timestepping(epsilon = epsilon * 4.0, L = 50, param_scaling = { 'std' : 40.0 })
hmc.sample(debug = False)#True)#False)#True)#False)#True)
#%%
hmc.derivative_check()
#%%
hmc.set_timestepping(epsilon = epsilon * 20, L = 75)
hmc.sample(debug = True)#False)#True)
#%%
hmc.print_current()
#%%
posterior = hmc.posterior_predictive(plot = False, lastN = 200, raw = True).reshape((30, 200))
#%%
posterior = posterior.reshape(-1, 200)
#%%
for r, datap, mean, stdd in zip(range(1, 31), data, posterior.mean(axis = 1), posterior.std(axis = 1)):
    print "{0} {1} {2:.2f} {3:.2f}".format(r, datap, mean, stdd)
#%%
hmc.posterior_predictive(plot = True, lastN = 200)
plt.title('Posterior predictive', fontsize = 72)
plt.xlabel('Mode', fontsize = 48)
plt.ylabel('Computed - Measured (khz)', fontsize = 48)
plt.tick_params(axis='y', which='major', labelsize=48)
plt.tick_params(axis='x', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches((24, 16))
plt.savefig('paper/ti/posteriorpredictive.png', dpi = 144)
plt.show()
#%%
hmc.save('/home/bbales2/modal/paper/ti/qs.csv')
#%%
import pickle

with open('paper/ti/ti_hmc_30_prior_0.3.pkl', 'w') as f:
    pickle.dump(hmc, f)
#%%
import pickle

with open('paper/ti/ti_hmc_30_prior_0.3.pkl', 'r') as f:
    hmc = pickle.load(f)
#%%
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(*hmc.format_samples()):
    plt.plot(data1[-2000:])
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
