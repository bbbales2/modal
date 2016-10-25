#%%
import numpy
import time
import scipy
import os
os.chdir('/home/pollock/modal')
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
X = 0.00550
Y = 0.01079
Z = 0.013025

## Dimensions for TF-2
#X = 0.007753#0.011959e1#
#Y = 0.009057#0.013953e1#
#Z = 0.013199#0.019976e1#

#sample mass
mass = 6.2593e-3 #mass in kg



#Sample density
#density = 4401.695921 #Ti-64-TF2
#density = 8700.0 #CMSX-4
density = (mass / (X*Y*Z))
 
c11 = 2.7
anisotropic = 1.1
c44 = 0.8
c12 = -(c44 * 2.0 / anisotropic - c11)

# Standard deviation around each mode prediction
std = 1.0

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

# data for sample 2M-B
#freqs = numpy.array([86.2498,
#109.5542,
#146.3892,
#156.1374,
#165.9964,
#166.7378,
#173.4004,
#181.1272,
#194.8802,
#204.6936,
#218.6092,
#224.8072,
#240.3284,
#249.4384,
#254.8852,
#257.6374,
#262.7806,
#281.0248,
#285.0576,
#286.7056,
#312.1016,
#315.6012,
#325.2092,
#326.6862,
#334.4188,
#337.9002,
#341.8525,
#343.7550,
#346.3570,
#349.8334])

#data for sample 2M-C
freqs = numpy.array([86.822,
109.347,
146.016,
155.950,
166.567,
167.098,
172.630,
181.351,
194.703,
204.054,
220.626,
225.696,
239.605,
248.984,
253.871,
258.744,
263.681,
281.705,
285.414,
287.231,
312.897,
316.988,
327.030,
329.371,
336.088,
338.079,
342.218,
344.306,
345.991,
346.751])

## Frequencies from 2M-F-Ave (30 modes)
#freqs = numpy.array([86.039,
#104.425,
#123.800,
#134.288,
#150.415,
#161.818,
#175.740,
#178.873,
#184.419,
#195.570,
#214.117,
#218.079,
#223.200,
#224.125,
#240.377,
#256.072,
#263.282,
#271.957,
#274.752,
#286.432,
#291.923,
#296.012,
#300.708,
#308.197,
#316.962,
#324.822,
#328.065,
#334.092,
#335.421,
#340.246])

# Ti-64-TF2 Test Data
#freqs = numpy.array([109.076,
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

data = (freqs * numpy.pi * 2000) ** 2 / 1e11

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
epsilon = 0.002

# Set this to true to debug the L and eps values
debug = False

#%%

#This block runs the HMC

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

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

dKdc11, _ = polybasis.buildKM(dCdc11, dp, pv, density)
dKdc12, _ = polybasis.buildKM(dCdc12, dp, pv, density)
dKdc44, _ = polybasis.buildKM(dCdc44, dp, pv, density)

def UgradU(q):
    c11, anisotropic, c44, std = q
    c12 = -(c44 * 2.0 / anisotropic - c11)
    #print q#X, Y, Z

    #tmp = time.time()
    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])
    #print "Basis build: ", time.time() - tmp

    K, M = polybasis.buildKM(C, dp, pv, density)
    #print 'Assemble: ', time.time() - tmp

    #tmp = time.time()
    #tmp = time.time()
    eigst, evecst = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))
    #print 'Eigs: ', time.time() - tmp
    #for e1, e2 in zip(eigst, data):
    #    print e1 - e2
    #print "\n".join(str(zip(eigst[6:], data)))
    #print "Eigs: ", time.time() - tmp

    eigst = eigst[:]
    evecst = evecst[:, :]

    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    dlpdl = (data - eigst) / std ** 2
    dlpdstd = sum((-std ** 2 + (eigst - data) **2) / std ** 3)

    dlpdl = numpy.array(dlpdl)

    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc44 = dlpdl.dot(dldc44)

    logp = sum(0.5 * (-((eigst - data) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

    dlpdc12tf = dlpdc12 * 2.0 * c44 / (anisotropic**2)
    return -logp, -numpy.array([dlpdc11 + dlpdc12, dlpdc12tf, dlpdc44 + dlpdc12 * -2 / anisotropic, dlpdstd])

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

c11s, anisotropics, c44s, stds = [numpy.array(aa)[-2000:] for aa in zip(*qs)]#
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(['c11', 'aniso ratio', 'c44', 'std deviation', '-logp'],
                      [c11s, anisotropics, c44s, stds, logps[-20000:]]):
    plt.plot(data1)
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 30)
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
c11s, anisotropics, c44s, stds = [numpy.array(ab)[-2000:] for ab in zip(*qs)]#

for name, data1 in zip(['c11', 'aniso ratio', 'c44', 'std dev'],
                      [c11s, anisotropics, c44s, stds]):
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 30)
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
