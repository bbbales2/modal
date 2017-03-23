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
X = 0.011959#0.007753
Y = 0.013953#0.009057
Z = 0.019976#0.013199

#Sample density
density = 8700.0#4401.695921 #Ti-64-TF2

c110 = 2.0
anisotropic0 = 2.0
c440 = 1.0
c120 = -(c440 * 2.0 / anisotropic0 - c110)

# Standard deviation around each mode prediction
std0 = 5.0

# Ti-64-TF2 Test Data
#data = numpy.array([109.076,
#136.503,
#144.899,
#184.926,
#188.476,
#195.562,
#199.246,
#208.460,
#231.220,
#232.630,
#239.057,
#241.684,
#242.159,
#249.891,
#266.285,
#272.672,
#285.217,
#285.670,
#288.796,
#296.976,
#301.101,
#303.024,
#305.115,
#305.827,
#306.939,
#310.428,
#318.000,
#319.457,
#322.249,
#323.464,
#324.702,
#334.687,
#340.427,
#344.087,
#363.798,
#364.862,
#371.704,
#373.248])

data = numpy.array([71.25925,
75.75875,
86.478,
89.947375,
111.150125,
112.164125,
120.172125,
127.810375,
128.6755,
130.739875,
141.70025,
144.50375,
149.40075,
154.35075,
156.782125,
157.554625,
161.0875,
165.10325,
169.7615,
173.44925,
174.11675,
174.90625,
181.11975,
182.4585,
183.98625,
192.68125,
193.43575,
198.793625,
201.901625,
205.01475,
206.619,
208.513875,
208.83525,
212.22525,
212.464125,
221.169625,
225.01225,
227.74775,
228.31175,
231.4265,
235.792875,
235.992375,
236.73675,
238.157625,
246.431125,
246.797125,
248.3185,
251.69425,
252.97225,
253.9795,
256.869875,
258.23825,
259.39025])[:30]

#data = numpy.array([  71.04149827,   82.74270332,   95.44792867,   99.16871298,
#        116.79412768,  121.52739404,  123.70569011,  137.16909551,
#        146.86716132,  149.21980347,  156.60206557,  163.68643926,
#        170.98528904,  175.59739109,  176.2592447 ,  185.9655138 ,
#        187.82188173,  189.33086255,  194.42119664,  197.07223083,
#        200.66774362,  206.21098658,  206.69925199,  211.46194875,
#        216.05725114,  217.84885983,  218.6616006 ,  223.55274233,
#        224.37361889,  227.74084719])
#
#data = numpy.array([  71.34475997,   83.2037272 ,   95.40214985,   98.63658633,
#        116.40953843,  122.360584  ,  124.1779111 ,  137.43584155,
#        147.47787637,  148.33982201,  156.24608537,  163.96310176,
#        170.84722907,  174.9822489 ,  176.97880246,  184.83976724,
#        187.6256789 ,  189.29388232,  193.65323046,  196.86083234,
#        200.15465607,  206.06214318,  206.77248337,  211.74757895,
#        215.51194915,  217.72926924,  218.69359608,  223.92339479,
#        224.68479063,  227.26967728,  229.25123788,  232.17777193,
#        235.96741787,  237.49517768,  245.38754132,  249.38440435,
#        251.50489501,  254.52515981,  255.56275932,  257.01470896,
#        266.3375151 ,  266.52476803,  269.55263391,  270.52162718,
#        272.66840067,  274.63510623,  275.27717333,  277.91917222,
#        281.6654003 ,  286.38803225])
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
              rotations = True,
              T = 1.0,
              stdMin = 0.0)

hmc.set_labels({ c11 : 'c11', anisotropic : 'a', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)
hmc.print_current()
hmc.sample(steps = 5, debug = True)
#%%
hmc.set_timestepping(epsilon = epsilon * 4.0, L = 50, param_scaling = { 'std' : 20.0 })
hmc.sample(debug = False)#False)#True)
#%%
hmc.derivative_check()
#%%
hmc.set_timestepping(epsilon = epsilon * 20, L = 75)
hmc.sample(debug = True)#False)#True)
#%%
hmc.print_current()
#%%
hmc.posterior_predictive(plot = False)
#%%
hmc.save('/home/bbales2/modal/paper/cmsx4/qs_norot.csv')
#%%
import pickle

with open('paper/cmsx4/hmc_30_noprior_norot.pkl', 'w') as f:
    pickle.dump(hmc, f)
#%%
import pickle

with open('paper/cmsx4/hmc_30_noprior.pkl') as f:
    hmc = pickle.load(f)

#%%
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(*hmc.format_samples()):
    plt.plot(data1[:])
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 24)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.show()
#%%
for name, data1 in zip(*hmc.format_samples()):
    data1 = data1[-4000:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.4f}, $\sigma$ = {2:0.4f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()
#%%
import sklearn.mixture

data = dict([(name, data[-4000:]) for name, data in zip(*hmc.format_samples())])

data = numpy.vstack((data['w_0'], data['x_0'], data['y_0'], data['z_0'])).T

gmm = sklearn.mixture.GMM(4)

gmm.fit(data)

print gmm.weights_
print gmm.means_
print gmm.covars_
#%%
angles = numpy.arccos(data[:, 0])

sins = numpy.sin(angles)

angles *= 180.0 / numpy.pi

for i in range(1, data.shape[1]):
    data[:, i] /= sins

data[:, 0] = angles

#%%

#with open('/home/bbales2/modal/paper/cmsx4/qs.csv', 'w') as f:
#    f.writeline("c11, c44, angle, ")
for name, data1 in zip(['angle', 'kx', 'ky', 'kz'], data.T):
    data1 = data1[-4000:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.4f}, $\sigma$ = {2:0.4f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()
#%%