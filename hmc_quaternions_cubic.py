#%%
import numpy
import time
import scipy
import os
os.chdir('/home/bbales2/modal')
import pyximport
pyximport.install(reload_support = True)

import polybasisqu
reload(polybasisqu)

#from rotations import symmetry
#from rotations import quaternion
#from rotations import inv_rotations

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

density = 8700.0#4401.695921#

# Dimensions -- watch the scaling
X = 0.011959#0.007753#
Y = 0.013953#0.009057#
Z = 0.019976#0.013199#

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
#%%
# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50
epsilon = 0.001

# Set this to true to debug the L and eps values
debug = False#True#

#%%
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

        print "Accepted ({0} accepts so far): {1}".format(len(accepts), ", ".join(["{0:.2f}".format(x) for x in current_q]))
    else:
        print "Rejected: ", ", ".join(["{0:.2f}".format(x) for x in current_q])

    qs.append(current_q.copy())
    print "Norm of rotation vector: ", numpy.linalg.norm(q[-4:])
    print "Energy change ({0} samples, {1} accepts): ".format(len(qs), len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K
    print "Epsilon: ", epsilon
#%%
c11s, anisotropics, c44s, stds, ws, xs, ys, zs  = [numpy.array(a) for a in zip(*qs)]#
import matplotlib.pyplot as plt

for name, data1 in zip(['c11', 'anisotropics', 'c44', 'stds', 'ws', 'xs', 'ys', 'zs'],
                      [c11s, anisotropics, c44s, stds, ws, xs, ys, zs]):
    plt.plot(data1)
    plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1[:200]), numpy.std(data1[:200])))
    plt.show()

#%%
import seaborn

c11s, anisotropics, c44s, stds, ws, xs, ys, zs  = [numpy.array(a)[-200:] for a in zip(*qs)]#
import matplotlib.pyplot as plt

for name, data1 in zip(['c11', 'anisotropics', 'c44', 'stds', 'ws', 'xs', 'ys', 'zs'],
                      [c11s, anisotropics, c44s, stds, ws, xs, ys, zs]):
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()


#%%
Ws = []
for w, x, y, z in zip(ws, xs, ys, zs):
    if numpy.abs(numpy.linalg.norm([w, x, y, z]) - 1.0) > 1e-10:
        print numpy.linalg.norm([w, x, y, z]) - 1.0
        1/0
    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    Ws.append(Q.T.dot([1.0, 0.0, 0.0]))

plt.plot(Ws)
plt.legend(['x-components', 'y-components', 'z-components'])
plt.show()
#%%
reload(polybasisqu)

dp1, pv1, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

for i in range(3):
    factors = numpy.array([X, Y, Z])
    factors[i] *= 1.0001
    Xt, Yt, Zt = factors
    dp2, pv2, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, Xt, Yt, Zt)

    ddps = [ddpdX, ddpdY, ddpdZ]
    dpvs = [dpvdX, dpvdY, dpvdZ]

    print (dp2[1, 1] - dp1[1, 1]) / (factors[i] - factors[i] / 1.0001)
    print ddps[i][1, 1]
    print (pv2[0, 0] - pv1[0, 0]) / (factors[i] - factors[i] / 1.0001)
    print dpvs[i][0, 0]
    print '----'

dp1, pv1, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, w, x, y, z)

K1, M1 = polybasisqu.buildKM(C, dp1, pv1, density)

a, b, y = 0.1, 0.2, 0.3

for i in range(3):
    factors = numpy.array([X, Y, Z])
    factors[i] *= 1.0001
    Xt, Yt, Zt = factors
    dp2, pv2, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, Xt, Yt, Zt)

    dKdX, dMdX = polybasisqu.buildKM(C, ddpdX, dpvdX, density)
    dKdY, dMdY = polybasisqu.buildKM(C, ddpdY, dpvdY, density)
    dKdZ, dMdZ = polybasisqu.buildKM(C, ddpdZ, dpvdZ, density)

    K2, M2 = polybasisqu.buildKM(C, dp2, pv2, density)

    ddps = [dKdX, dKdY, dKdZ]
    dpvs = [dMdX, dMdY, dMdZ]

    print (K2[494, 486] - K1[494, 486]) / (factors[i] - factors[i] / 1.0001)
    print ddps[i][494, 486]
    print (M2[0, 12] - M1[0, 12]) / (factors[i] - factors[i] / 1.0001)
    print dpvs[i][0, 12]
    print '----'
#%%
reload(polybasisqu)

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

C1, dCdw1, dCdx1, dCdy1, dCdz1, K = polybasisqu.buildRot(C, w, x, y, z)

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

    #tmp = time.time()
dKdw1, _ = polybasisqu.buildKM(dCdw1, dp, pv, density)
dKdx1, _ = polybasisqu.buildKM(dCdx1, dp, pv, density)
dKdy1, _ = polybasisqu.buildKM(dCdy1, dp, pv, density)
dKdz1, _ = polybasisqu.buildKM(dCdz1, dp, pv, density)

dKdc111, _ = polybasisqu.buildKM(dCdc11, dp, pv, density)
dKdc121, _ = polybasisqu.buildKM(dCdc12, dp, pv, density)
dKdc441, _ = polybasisqu.buildKM(dCdc44, dp, pv, density)

dKdX1, dMdX1 = polybasisqu.buildKM(C1, ddpdX, dpvdX, density)
dKdY1, dMdY1 = polybasisqu.buildKM(C1, ddpdY, dpvdY, density)
dKdZ1, dMdZ1 = polybasisqu.buildKM(C1, ddpdZ, dpvdZ, density)

K1, M1 = polybasisqu.buildKM(C1, dp, pv, density)

w = 0.5
x = 0.3
y = 0.1
z = numpy.sqrt(1 - w**2 - x**2 - y**2)

for i in range(4):
    #factors = numpy.array([c11, c12, c44])
    factors = numpy.array([w, x, y, z])
    factors[i] *= 1.0001
    #c11t, c12t, c44t = factors
    wt, xt, yt, zt = factors
    c11t, c12t, c44t = c11, c12, c44

    C = numpy.array([[c11t, c12t, c12t, 0, 0, 0],
                     [c12t, c11t, c12t, 0, 0, 0],
                     [c12t, c12t, c11t, 0, 0, 0],
                     [0, 0, 0, c44t, 0, 0],
                     [0, 0, 0, 0, c44t, 0],
                     [0, 0, 0, 0, 0, c44t]])

    C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, wt, xt, yt, zt)

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

        #tmp = time.time()
    dKdw, _ = polybasisqu.buildKM(dCdw, dp, pv, density)
    dKdx, _ = polybasisqu.buildKM(dCdx, dp, pv, density)
    dKdy, _ = polybasisqu.buildKM(dCdy, dp, pv, density)
    dKdz, _ = polybasisqu.buildKM(dCdz, dp, pv, density)

    dKdc11, _ = polybasisqu.buildKM(dCdc11, dp, pv, density)
    dKdc12, _ = polybasisqu.buildKM(dCdc12, dp, pv, density)
    dKdc44, _ = polybasisqu.buildKM(dCdc44, dp, pv, density)

    dKdX, dMdX = polybasisqu.buildKM(C, ddpdX, dpvdX, density)
    dKdY, dMdY = polybasisqu.buildKM(C, ddpdY, dpvdY, density)
    dKdZ, dMdZ = polybasisqu.buildKM(C, ddpdZ, dpvdZ, density)

    K, M = polybasisqu.buildKM(C, dp, pv, density)

    #ders = [dCdw, dCdx, dCdy, dCdz]
    ders = [dKdw, dKdx, dKdy, dKdz]
    #ders = [dKdc111, dKdc121, dKdc441]

    print ((K - K1) / (factors[i] - factors[i] / 1.0001))
    print ders[i]
    print '--'