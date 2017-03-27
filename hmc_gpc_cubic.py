#%%
import numpy
import time
import scipy
import os
os.chdir('/home/bbales2/modal')
import pyximport
pyximport.install(reload_support = True)

import polybasis
reload(polybasis)

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

## Dimensions for TF-2
X = 0.007753#0.011959e1#
Y = 0.009057#0.013953e1#
Z = 0.013199#0.019976e1#

#sample mass

#Sample density
density = 4401.695921 #Ti-64-TF2
#density = 8700.0 #CMSX-4
#density = (mass / (X*Y*Z))

c11 = 2.0
anisotropic = 1.0
c44 = 1.0
c12 = -(c44 * 2.0 / anisotropic - c11)

# Standard deviation around each mode prediction
std = 1.0

# Rotations
a = 0.0
b = 0.0
y = 0.0

# These are the sampled modes in khz
# data for sample 2M-A

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
373.248])

#data = (freqs * numpy.pi * 2000) ** 2 / 1e11

qs = []
logps = []
accepts = []

current_q = numpy.array([c11, anisotropic, c44, std])

accepts.append(current_q)
#%%

# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50
# start epsilon at .0001 and try larger values like .0005 after running for a while
# epsilon is timestep, we want to make as large as possibe, wihtout getting too many rejects
epsilon = 0.0005

# Set this to true to debug the L and eps values
debug = False

#%%
import polybasisqu
import sys
sys.path.append('/home/bbales2/gpc')
import gpc

def func(c11, anisotropic, c44):
    c12 = -(c44 * 2.0 / anisotropic - c11)

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

func2 = lambda c11, a, c44 : func(c11, 1.0, c44)

minc11 = 1.0#1.6
maxc11 = 2.0#1.8
mina = 1.0
maxa = 1.5
minc44 = 0.3#0.448
maxc44 = 0.8#0.452

hd = gpc.GPC(5, func2, [('n', (2.0, 0.5), 3),
                       ('u', (mina, maxa), 5),
                       ('u', (minc44, maxc44), 5)])

#%%
#This block runs the HMC

def UgradU(q):
    c11, anisotropic, c44, std = q

    anisotropic = 1.0

    try:
        e = hd.approx(c11, anisotropic, c44)
        dedc11 = hd.approxd(0, c11, anisotropic, c44)
        deda = hd.approxd(1, c11, anisotropic, c44)
        dedc44 = hd.approxd(2, c11, anisotropic, c44)
    except Exception as e:
        e = numpy.nan
        dedc11 = [numpy.nan] * len(data)
        deda = [numpy.nan] * len(data)
        dedc44 = [numpy.nan] * len(data)

    dlpde = (data - e) / std ** 2
    dlpdstd = sum((-std ** 2 + (e - data) **2) / std ** 3)

    #dlpde = numpy.array(dlpde)

    dlpdc11 = dlpde.dot(dedc11)
    dlpda = dlpde.dot(deda)
    dlpdc44 = dlpde.dot(dedc44)

    logp = sum(0.5 * (-((e - data) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

    return -logp, -numpy.array([dlpdc11, 0.0, dlpdc44, dlpdstd])

#%%
d = 0.00001
q = numpy.array([1.6, 1.0, 0.7, 1.0])

nlogp, dnlogp = UgradU(q)

for i in range(3):
    q_ = q.copy()

    q_[i] += d

    nlogp_, _ = UgradU(q_)

    print dnlogp[i], (nlogp_ - nlogp) / d
#%%
c11 = 1.7
anisotropic = 1.0
c44 = 0.5

qs = []
logps = []
accepts = []

current_q = numpy.array([c11, anisotropic, c44, std])

accepts.append(current_q)

debug = False

while True:
    q = current_q.copy()
    p = numpy.random.randn(len(q)) # independent standard normal variates

    current_p = p
    # Make a half step for momentum at the beginning
    U, gradU = UgradU(q)
    p = p - epsilon * gradU / 2

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        #q[-3:] = inv_rotations.qu2eu(symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(inv_rotations.eu2qu(q[-3:]))))
        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradU = UgradU(q)
            p = p - epsilon * gradU

        #print 'hi'
        if debug:
            print "New q: ", q
            print "H (constant or decreasing): ", U + sum(p ** 2) / 2, U, sum(p **2) / 2.0
            print ""

    U, gradU = UgradU(q)
    # Make a half step for momentum at the end.
    p = p - epsilon * gradU / 2

    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    UC, gradUC = UgradU(current_q)
    current_U = UC
    current_K = sum(current_p ** 2) / 2
    proposed_U = U
    proposed_K = sum(p ** 2) / 2

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    dQ = current_U - proposed_U + current_K - proposed_K

    logps.append(UC)

    if numpy.random.rand() < min(1.0, numpy.exp(dQ)) and not numpy.isnan(proposed_U):
        current_q = q # accept

        accepts.append(len(qs) - 1)

        print "Accepted ({0} accepts so far): {1}".format(len(accepts), current_q)
    else:
        print "Rejected: ", current_q

    qs.append(q.copy())
    print "Energy change ({0} samples, {1} accepts): ".format(len(qs), len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K


#%%
# Save samples (qs)
# First argument is filename

import os
import tempfile
import datetime

_, filename = tempfile.mkstemp(prefix = "data_{0}_".format(datetime.datetime.now().strftime("%Y-%m-%d")), suffix = ".txt", dir = os.getcwd())
numpy.savetxt(filename, qs, header = 'c11 anisotropic c44 std')
#%%
# This block does the plotting

c11s, anisotropics, c44s, stds = [numpy.array(a) for a in zip(*qs)]#
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(['c11', 'anisotropic ratio', 'c44', 'std deviation', '-logp'],
                      [c11s, anisotropics, c44s, stds, logps]):
    plt.plot(data1)
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 24)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.show()
    #seaborn.distplot(d[-650:], kde = False, fit = scipy.stats.norm)
    #plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    #plt.show()

#plt.plot(as_, ys_)
#plt.ylabel('eu[2]s')
#plt.xlabel('eu[0]s')
#plt.show()
#%%
import seaborn
c11s, anisotropics, c44s, stds = [numpy.array(a)[-1500:] for a in zip(*qs)]#

for name, data1 in zip(['C11', 'A Ratio', 'C44', 'std dev'],
                      [c11s, anisotropics, c44s, stds]):
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()
    #seaborn.distplot(d[-650:], kde = False, fit = scipy.stats.norm)
    #plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    #plt.show()

#%%

while 1:
    U, gradU = UgradU(current_q)

    current_q += 0.0001 * gradU
#%%
# Forward problem

# This snippet is helpful to test the last accepted sample
c11, anisotropic, c44, std = qs[accepts[-1]]

c12 = -(c44 * 2.0 / anisotropic - c11)

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                 [c12, c11, c12, 0, 0, 0],
                 [c12, c12, c11, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c44, 0],
                 [0, 0, 0, 0, 0, c44]])

K, M = polybasis.buildKM(C, dp, pv, density)
eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))

print "computed, accepted"
for e1, dat in zip(eigs, data):
    print "{0:0.3f} {1:0.3f}".format(e1, dat)
