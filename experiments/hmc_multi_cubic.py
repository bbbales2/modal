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

#from rotations import symmetry
#from rotations import quaternion
#from rotations import inv_rotations

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 8

# Dimensions for sample 2M-A (units of meters)
Xs = [0.007753, 0.007722]#0.00550 (2M-A)
Ys = [0.009057, 0.009032]#0.01079 (2M-A)
Zs = [0.013199, 0.013213]#0.01302 (2M-A)

S = len(Xs) # Number of samples

## Dimensions for TF-2
#X = 0.007753#0.011959e1#
#Y = 0.009057#0.013953e1#
#Z = 0.013199#0.019976e1#

#sample mass
masses = [4.0795e-3, 4.0487e-3]#6.2203e-3 #mass in kg


#Sample density
#density = 4401.695921 #Ti-64-TF2
#density = 8700.0 #CMSX-4
densities = [mass / (X * Y * Z) for mass, X, Y, Z in zip(masses, Xs, Ys, Zs)]

c11 = 2.0
anisotropic = 1.0
c44 = 1.0
c12 = -(c44 * 2.0 / anisotropic - c11)

# Standard deviation around each mode prediction
std = 5.0

# Rotations
a = 0.0
b = 0.0
y = 0.0

# These are the sampled modes in khz
# data for sample 2M-A
#freqs = numpy.array([86.916,
#108.277,
#146.582,
#155.722,
#166.274,
#167.510,
#173.016,
#181.229,
#195.209,
#204.479,
#218.063,
#225.679,
#240.271,
#251.107,
#254.079,
#258.857,
#264.639,
#281.978,
#286.834,
#287.739,
#311.693,
#318.010,
#327.643,
#328.771,
#336.807,
#338.585,
#342.069,
#344.734,
#346.879,
#350.192
#])

# Ti-64-TF2 Test Data
freqs = [numpy.array([109.076,
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
373.248]),
numpy.array([109.702,
137.103,
145.614,
185.868,
189.735,
196.581,
200.868,
209.287,
231.463,
233.534,
239.976,
241.539,
242.556,
250.643,
265.716,
274.081,
286.145,
287.189,
289.480,
298.727,
301.739,
304.506,
306.816,
308.469,
309.756,
311.606,
320.335,
320.665,
323.950,
325.469,
326.218,
336.009,
341.652,
346.285,
365.315,
366.239,
373.038,
374.744,
380.775,
381.627,
385.013,
387.057,
390.899,
393.962])]

data = freqs

Modes = max([len(a) for a in data])
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
epsilon = 0.001

# Set this to true to debug the L and eps values
debug = False

#%%

#This block runs the HMC

dps = []
pvs = []

dKdc11s = []
dKdc12s = []
dKdc44s = []

dCdc11 = numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

dCdc12 = numpy.array([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

dCdc44 = numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

for density, X, Y, Z in zip(densities, Xs, Ys, Zs):
    dp, pv, _, _, _, _, _, _ = polybasis.build(N, X, Y, Z)

    dps.append(dp)
    pvs.append(pv)

    dKdc11, _ = polybasis.buildKM(dCdc11, dp, pv, density)
    dKdc12, _ = polybasis.buildKM(dCdc12, dp, pv, density)
    dKdc44, _ = polybasis.buildKM(dCdc44, dp, pv, density)

    dKdc11s.append(dKdc11)
    dKdc12s.append(dKdc12)
    dKdc44s.append(dKdc44)

def UgradU(q):
    c11, anisotropic, c44, std = q
    c12 = -(c44 * 2.0 / anisotropic - c11)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    logp_total = 0.0
    dlogp_total = numpy.zeros(4)

    for s in range(S):
        K, M = polybasis.buildKM(C, dp, pv, density)

        eigst, evecst = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data[s]) - 1))

        eigst = eigst[:]
        evecst = evecst[:, :]

        dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
        dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
        dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

        freqst = numpy.sqrt(eigst * 1e11) / (numpy.pi * 2000)#(freqs * numpy.pi * 2000) ** 2 / 1e11

        dfreqsdl = 0.5e11 / (numpy.sqrt(eigst * 1e11) * numpy.pi * 2000)

        dlpdfreqs = (data[s] - freqst) / std ** 2
        dlpdstd = sum((-std ** 2 + (freqst - data[s]) **2) / std ** 3)

        #print q
        #print eigst
        #print freqst

        dlpdl = dlpdfreqs * dfreqsdl
        #print dlpdl

        dlpdc11 = dlpdl.dot(dldc11)
        dlpdc12 = dlpdl.dot(dldc12)
        dlpdc44 = dlpdl.dot(dldc44)

        logp = sum(0.5 * (-((freqst - data[s]) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

        dlpdc12tf = dlpdc12 * 2.0 * c44 / (anisotropic**2)

        logp_total += logp
        dlogp_total += numpy.array([dlpdc11 + dlpdc12, dlpdc12tf, dlpdc44 + dlpdc12 * -2 / anisotropic, dlpdstd])

    return -logp_total, -dlogp_total

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

    if numpy.random.rand() < min(1.0, numpy.exp(dQ)):
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
numpy.savetxt("/home/bbales2/modal/paper/ti/qs.csv", qs, delimiter = ",", comments = "", header = "c11, anisotropic, c44, std")
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
#N = 9
# This snippet is helpful to test the last accepted sample
for s, density, X, Y, Z in zip(range(S), densities, Xs, Ys, Zs):
    c11, anisotropic, c44, std = current_q#qs[accepts[-1]]
    #c11, anisotropic, c44, std = numpy.array([ 1.72,  1.00,  0.44,  1.67037655])

    c12 = -(c44 * 2.0 / anisotropic - c11)

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    K, M = polybasis.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data[s]) - 1))
    freqst = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

    print numpy.mean(freqst - data[s])
    print numpy.std(freqst - data[s])

    #print "computed, accepted"
    #for e1, dat in zip(freqst, data):
    #    print "{0:0.3f} {1:0.3f}".format(e1, dat)
