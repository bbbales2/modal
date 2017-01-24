#%%
import numpy
import time
import scipy
import os
os.chdir('/home/bbales2/modal')
import pyximport
import seaborn
pyximport.install(reload_support = True)

import polybasisqu
reload(polybasisqu)

#from rotations import symmetry
#from rotations import quaternion
#from rotations import inv_rotations

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 12

density = 8700.0  #4401.695921#

# Dimensions -- watch the scaling
X = .011959  #0.007753#
Y = .013953  #0.009057#
Z = .019976  #0.013199#

c11 = 2.0
anisotropic = 2.0
c44 = 1.0
c12 = -(c44 * 2.0 / anisotropic - c11)

# Standard deviation around each mode prediction
std = 1.0

# Rotations
w = 1.0
x = 0.0
y = 0.0
z = 0.0

# These are the sampled modes in khz

# Frequencies from SXSA
freqs = numpy.array([
68.066,
87.434,
104.045,
105.770,
115.270,
122.850,
131.646,
137.702,
139.280,
149.730,
156.548,
156.790,
169.746,
172.139,
173.153,
178.047,
183.433,
188.288,
197.138,
197.869,
198.128,
203.813,
206.794,
212.173,
212.613,
214.528,
215.840,
221.452,
227.569,
232.430])



# Frequencies for CMSX-4
freqs = numpy.array([71.25925,
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
259.39025])

data = (freqs * numpy.pi * 2000) ** 2 / 1e11

qs = []
logps = []
accepts = []

current_q = numpy.array([c11, anisotropic, c44, std, w, x, y, z])
#current_q = [.243914164e+00, 2.87679060e+00, .131354678e+00, 6.10899449e-02, 9.88260173e-01, 3.40517820e-05, -1.15781363e-02, -1.52190782e-01]

#%%
# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50

# start epsilon at .0001 and try larger values like .0005 after running for a while
epsilon = 0.00025

# Set this to true to debug the L and eps values
debug = False#True#

#%%
# This is for running the HMC
def UgradU(q):
    c11, anisotropic, c44, std, w, x, y, z = q
    c12 = -(c44 * 2.0 / anisotropic - c11)
    #print q#X, Y, Z

    #tmp = time.time()
    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])
    #print "Basis build: ", time.time() - tmp

    #tmp = time.time()
    C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, w, x, y, z)

    dCdc11 = K.dot(numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

    dCdc12 = K.dot(numpy.array([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))

    dCdc44 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).dot(K.T))
    #print "Rotation time: ", time.time() - tmp

    #tmp = time.time()
    dKdw, _ = polybasisqu.buildKM(dCdw, dp, pv, density)
    dKdx, _ = polybasisqu.buildKM(dCdx, dp, pv, density)
    dKdy, _ = polybasisqu.buildKM(dCdy, dp, pv, density)
    dKdz, _ = polybasisqu.buildKM(dCdz, dp, pv, density)

    dKdc11, _ = polybasisqu.buildKM(dCdc11, dp, pv, density)
    dKdc12, _ = polybasisqu.buildKM(dCdc12, dp, pv, density)
    dKdc44, _ = polybasisqu.buildKM(dCdc44, dp, pv, density)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    print K.shape
    #print 'Assemble: ', time.time() - tmp

    #tmp = time.time()
    #tmp = time.time()
    eigst, evecst = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))
    #print 'Eigs: ', time.time() - tmp
    #for e1, e2 in zip(eigst, data):
    #    print e1, e2
    #print "\n".join(str(zip(eigst[6:], data)))
    #print "Eigs: ", time.time() - tmp

    eigst = eigst[:]
    evecst = evecst[:, :]

    dldw = numpy.array([evecst[:, i].T.dot(dKdw.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldx = numpy.array([evecst[:, i].T.dot(dKdx.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldy = numpy.array([evecst[:, i].T.dot(dKdy.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldz = numpy.array([evecst[:, i].T.dot(dKdz.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    dlpdl = (data - eigst) / std ** 2
    dlpdstd = sum((-std ** 2 + (eigst - data) **2) / std ** 3)

    dlpdl = numpy.array(dlpdl)

    dlpdw = dlpdl.dot(dldw)
    dlpdx = dlpdl.dot(dldx)
    dlpdy = dlpdl.dot(dldy)
    dlpdz = dlpdl.dot(dldz)
    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc44 = dlpdl.dot(dldc44)

    logp = sum(0.5 * (-((eigst - data) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

    dlpdc12tf = dlpdc12 * 2.0 * c44 / (anisotropic**2)
    return -logp, -numpy.array([dlpdc11 + dlpdc12, dlpdc12tf, dlpdc44 + dlpdc12 * -2 / anisotropic, dlpdstd, dlpdw, dlpdx, dlpdy, dlpdz])

while True:
    q = current_q.copy()
    p = numpy.random.randn(len(q)) # independent standard normal variates
    p[-4:] -= numpy.outer(q[-4:], q[-4:]).dot(p[-4:])

    current_p = p.copy()
    #print current_p, current_q
    # Make a half step for momentum at the beginning
    U, gradU = UgradU(q)
    p = p - epsilon * gradU / 2
    p[-4:] -= numpy.outer(q[-4:], q[-4:]).dot(p[-4:])

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q[:-4] = q[:-4] + epsilon * p[:-4]

        alpha = numpy.linalg.norm(p[-4:])
        m1 = numpy.array([[1.0, 0.0],
                          [0.0, 1 / alpha]])

        m2 = numpy.array([[numpy.cos(alpha * epsilon), -numpy.sin(alpha * epsilon)],
                          [numpy.sin(alpha * epsilon), numpy.cos(alpha * epsilon)]])

        m3 = numpy.array([[1.0, 0.0],
                          [0.0, alpha]])

        xv = numpy.array([q[-4:], p[-4:]]).T.dot(m1.dot(m2.dot(m3)))

        q[-4:] = xv[:, 0]
        p[-4:] = xv[:, 1]

        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradU = UgradU(q)
            p = p - epsilon * gradU
            p[-4:] -= numpy.outer(q[-4:], q[-4:]).dot(p[-4:])

        #if numpy.isnan(U):
        #    1/0
        #if debug:
        #    print "New q: ", q
        #    print "H (constant or decreasing): ", U + sum(p ** 2) / 2
        #    print ""

    U, gradU = UgradU(q)
    # Make a half step for momentum at the end.
    p = p - epsilon * gradU / 2
    p[-4:] -= numpy.outer(q[-4:], q[-4:]).dot(p[-4:])

    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    # Evaluate potential and kinetic energies at start and end of trajectory
    #print current_p, current_q
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

        print "Accepted ({0} accepts so far): {1}".format(len(accepts), ", ".join(["{0:.4f}".format(qq) for qq in current_q]))
    else:
        print "Rejected: ", ", ".join(["{0:.4f}".format(qq) for qq in current_q])

    qs.append(current_q.copy())
    print "Norm of rotation vector: ", numpy.linalg.norm(q[-4:])
    print "Energy change ({0} samples, {1} accepts): ".format(len(qs), len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K
    print "Epsilon: ", epsilon
#%%

import matplotlib.pyplot as plt

for N in [8, 10, 12]:
    current_q = [.243914164e+00, 2.87679060e+00, .131354678e+00, 6.10899449e-02, 9.88260173e-01, 3.40517820e-05, -1.15781363e-02, -1.52190782e-01]

    xs = numpy.linspace(2.5 * -1.15781363e-02, 2.5 * 1.20451685e-02, 20)
    Us = []
    for i in range(len(xs)):
        current_q[6] = xs[i]

        current_q[4] = numpy.sqrt(1.0 - current_q[5]**2 - current_q[6]**2 - current_q[7]**2)

        U, gradU = UgradU(current_q)

        Us.append(U)
        print "{0}/{1}".format(i, len(xs))

    plt.plot(xs, Us)
    plt.title("N = {0}".format(N))
    plt.xlabel('y')
    plt.ylabel('-logp')
    plt.show()

#%%
numpy.savetxt("/home/bbales2/modal/paper/cmsx4/qs.csv", qs, delimiter = ",", comments = "", header = "c11, anisotropic, c44, std, ws, xs, ys, zs")
#%%
# Save samples (qs)
# First argument is filename

import os
import tempfile
import datetime

_, filename = tempfile.mkstemp(prefix = "data_{0}_".format(datetime.datetime.now().strftime("%Y-%m-%d")), suffix = ".txt", dir = os.getcwd())
numpy.savetxt(filename, qs, header = 'c11 anisotropic c44 std w x y z')
#%%
# This is for plotting the trajectory of the samples through space
c11s, anisotropics, c44s, stds, ws, xs, ys, zs  = [numpy.array(a)[-8000:] for a in zip(*qs)]#
import matplotlib.pyplot as plt

for name, data1 in zip(['c11', 'anisotropics', 'c44', 'stds', 'ws', 'xs', 'ys', 'zs'],
                      [c11s, anisotropics, c44s, stds, ws, xs, ys, zs]):
    plt.plot(data1)
    plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    plt.show()

#%%
# This is for plotting distributions of the parameters

c11s, anisotropics, c44s, stds, ws, xs, ys, zs  = [numpy.array(a) for a in zip(*qs)]#
import matplotlib.pyplot as plt

for name, data1 in zip(['c11', 'anisotropics', 'c44', 'stds', 'ws', 'xs', 'ys', 'zs'],
                      [c11s, anisotropics, c44s, stds, ws, xs, ys, zs]):
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.4f}, $\sigma$ = {2:0.4f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()

#%%
# This is for seeing the passive rotation applied to a reference x-axis (to see what things converged too)
Ws = []
for w, x, y, z in zip(ws, xs, ys, zs):
    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    Ws.append(Q.T.dot([1.0, 0.0, 0.0]))

plt.plot(Ws)
plt.legend(['x-components', 'y-components', 'z-components'])
plt.show()
#%%#%%
# Forward problem

# This snippet is helpful to test the last accepted sample
#c11, anisotropic, c44, std, w, x, y, z = qs[accepts[-1]]
#current_q = [  2.43914164e+00,   2.87679060e+00,   1.31354678e+00,   6.10899449e-02, 9.88260173e-01,   3.40517820e-05,  -1.15781363e-02,  -1.52190782e-01]
#current_q = [  2.44439470e+00,   2.87642565e+00,   1.31315193e+00,   6.09557485e-02, 9.88248568e-01,  -4.77945975e-04,   0.0,  -1.52234546e-01]
current_q = [  2.43914164e+00,   2.87679060e+00,   1.31354678e+00,   6.10899449e-02, 9.88260173e-01,   3.40517820e-05,  0.0,  -1.52190782e-01]
current_q[4] = numpy.sqrt(1.0 - current_q[5]**2 - current_q[6]**2 - current_q[7]**2)

c11, anisotropic, c44, std, w, x, y, z = current_q
w = 1.0
x = 0.0
y = 0.0
z = 0.0

#%%
from rotations import quaternion

tmp = quaternion.Quaternion([w, x, y, z])
rot = quaternion.Quaternion([0.86835826819129436, -0.34803311754280902, -0.31462551582333742, -0.16074094671286512])
w, x, y, z = (rot * tmp).wxyz

#%%

c12 = -(c44 * 2.0 / anisotropic - c11)

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                 [c12, c11, c12, 0, 0, 0],
                 [c12, c12, c11, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c44, 0],
                 [0, 0, 0, 0, 0, c44]])

C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, w, x, y, z)
K, M = polybasisqu.buildKM(C, dp, pv, density)
eigs2, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))

print "computed, accepted"
for e1, dat in zip(eigs2, data):
    print "{0:0.5f} {1:0.3f}".format(e1, dat)
#%%

print "minimum (y = -0.015), y = 0.0, measured, error vs. y = -0.015, error vs. y = 0.0"
for e1, e2, dat in zip(eigs, eigs2, data):
    print "{0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f}".format(e1, e2, dat, numpy.abs(e1 - dat), numpy.abs(e2 - dat))
