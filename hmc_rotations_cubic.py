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
N = 10

density = 8700.0#4401.695921#

# Dimensions -- watch the scaling
X = 0.011959#0.007753#
Y = 0.013953#0.009057#
Z = 0.019976#0.013199#

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

current_q = numpy.array([c11, anisotropic, c44, std, X, Y, Z, a, b, y])
#%%
# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50
epsilon = 0.0001

# Set this to true to debug the L and eps values
debug = False#True#

#%%
def UgradU(q):
    c11, anisotropic, c44, std, X, Y, Z, a, b, y = q
    c12 = -(c44 * 2.0 / anisotropic - c11)
    #print q#X, Y, Z

    #tmp = time.time()
    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])
    #print "Basis build: ", time.time() - tmp

    #tmp = time.time()
    C, dCda, dCdb, dCdy, K = polybasis.buildRot(C, a, b, y)

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
    dKda, _ = polybasis.buildKM(dCda, dp, pv, density)
    dKdb, _ = polybasis.buildKM(dCdb, dp, pv, density)
    dKdy, _ = polybasis.buildKM(dCdy, dp, pv, density)

    dKdc11, _ = polybasis.buildKM(dCdc11, dp, pv, density)
    dKdc12, _ = polybasis.buildKM(dCdc12, dp, pv, density)
    dKdc44, _ = polybasis.buildKM(dCdc44, dp, pv, density)

    dKdX, dMdX = polybasis.buildKM(C, ddpdX, dpvdX, density)
    dKdY, dMdY = polybasis.buildKM(C, ddpdY, dpvdY, density)
    dKdZ, dMdZ = polybasis.buildKM(C, ddpdZ, dpvdZ, density)

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

    dlda = numpy.array([evecst[:, i].T.dot(dKda.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldb = numpy.array([evecst[:, i].T.dot(dKdb.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldy = numpy.array([evecst[:, i].T.dot(dKdy.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldX = numpy.array([evecst[:, i].T.dot((dKdX - eigst[i] * dMdX).dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldY = numpy.array([evecst[:, i].T.dot((dKdY - eigst[i] * dMdY).dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldZ = numpy.array([evecst[:, i].T.dot((dKdZ - eigst[i] * dMdZ).dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc11 = numpy.array([evecst[:, i].T.dot(dKdc11.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc12 = numpy.array([evecst[:, i].T.dot(dKdc12.dot(evecst[:, i])) for i in range(evecst.shape[1])])
    dldc44 = numpy.array([evecst[:, i].T.dot(dKdc44.dot(evecst[:, i])) for i in range(evecst.shape[1])])

    dlpdl = (data - eigst) / std ** 2
    dlpdstd = sum((-std ** 2 + (eigst - data) **2) / std ** 3)

    dlpdl = numpy.array(dlpdl)

    dlpda = dlpdl.dot(dlda)
    dlpdb = dlpdl.dot(dldb)
    dlpdy = dlpdl.dot(dldy)
    dlpdX = dlpdl.dot(dldX)
    dlpdY = dlpdl.dot(dldY)
    dlpdZ = dlpdl.dot(dldZ)
    dlpdc11 = dlpdl.dot(dldc11)
    dlpdc12 = dlpdl.dot(dldc12)
    dlpdc44 = dlpdl.dot(dldc44)

    logp = sum(0.5 * (-((eigst - data) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

    dlpdc12tf = dlpdc12 * 2.0 * c44 / (anisotropic**2)
    return -logp, -numpy.array([dlpdc11 + dlpdc12, dlpdc12tf, dlpdc44 + dlpdc12 * -2 / anisotropic, dlpdstd, dlpdX, dlpdY, dlpdZ, dlpda, dlpdb, dlpdy])

while True:
    q = current_q
    p = numpy.random.randn(len(q)) # independent standard normal variates

    current_p = p
    # Make a half step for momentum at the beginning
    U, gradU = UgradU(q)
    p = p - epsilon * gradU / 2

    p[-6:] = 0

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        #q[-3:] = inv_rotations.qu2eu(symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(inv_rotations.eu2qu(q[-3:]))))
        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradU = UgradU(q)
            p = p - epsilon * gradU

            p[-6:] = 0

        if debug:
            print "New q: ", q
            print "H (constant or decreasing): ", U + sum(p ** 2) / 2
            print ""

    U, gradU = UgradU(q)
    # Make a half step for momentum at the end.
    p = p - epsilon * gradU / 2
    p[-6:] = 0

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

    qs.append(q)
    logps.append(UC)

    if numpy.random.rand() < min(1.0, numpy.exp(dQ)):
        current_q = q # accept

        accepts.append(len(qs) - 1)

        print "Accepted ({0} accepts so far): {1}".format(len(accepts), current_q)
    else:
        print "Rejected: ", current_q

    print "Energy change ({0} samples, {1} accepts): ".format(len(qs), len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K
#%%
c11s, anisotropics, c44s, stds, Xs, Ys, Zs, as_, bs_, ys_  = [numpy.array(a) for a in zip(*[qs[i] for i in accepts])]#
import matplotlib.pyplot as plt

for name, data1 in zip(['c11', 'anisotropic ratio', 'c44', 'stds', 'Xs', 'Ys', 'Zs', 'eu[0]', 'eu[1]', 'eu[2]', '-log probs'],
                      [c11s, anisotropics, c44s, stds, Xs, Ys, Zs, as_, bs_, ys_, logps]):
    plt.plot(data1)
    plt.title('{0}, $\mu$ = {1:.3f}, $\sigma$ = {2:.3f}'.format(name, numpy.mean(data1[-400:]), numpy.std(data1[-400:])), fontsize = 24)
    plt.show()
    #seaborn.distplot(d[-650:], kde = False, fit = scipy.stats.norm)
    #plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    #plt.show()

#plt.plot(as_, ys_)
#plt.ylabel('eu[2]s')
#plt.xlabel('eu[0]s')
#plt.show()
#%%
c11s, anisotropics, c44s, stds, Xs, Ys, Zs, as_, bs_, ys_  = [numpy.array(a)[-400:] for a in zip(*[qs[i] for i in accepts])]#
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(['c11', 'anisotropic ratio', 'c44', 'stds', 'Xs', 'Ys', 'Zs', 'eu[0]', 'eu[1]', 'eu[2]', '-log probs'],
                      [c11s, anisotropics, c44s, stds, Xs, Ys, Zs, as_, bs_, ys_, logps]):
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()
    #seaborn.distplot(d[-650:], kde = False, fit = scipy.stats.norm)
    #plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    #plt.show()

#plt.plot(as_, ys_)
#plt.ylabel('eu[2]s')
#plt.xlabel('eu[0]s')
#plt.show()
#%%
rs = []
for a, b, y in zip(as_, bs_, ys_):
    s = numpy.sin((a, b, y))
    c = numpy.cos((a, b, y))

    Q = numpy.zeros((3, 3))

    Q = numpy.array([[c[0] * c[2] - s[0] * c[1] * s[2], s[0] * c[2] + c[0] * c[1] * s[2], s[1] * s[2]],
                     [-c[0] * s[2] - s[0] * c[1] * c[2], -s[0] * s[2] + c[0] * c[1] * c[2], s[1] * c[2]],
                     [s[0] * s[1], -c[0] * s[1], c[1]]])

    tmp = Q.T.dot([1.0, 0.0, 0.0])

    rs.append(tmp)

rs = numpy.array(rs)

plt.plot(rs[:, 0], 'r')
plt.plot(rs[:, 1], 'g')
plt.plot(rs[:, 2], 'b')
plt.legend(['x-component', 'y-component', 'z-component'])
plt.show()
#%%
c11, anisotropic, c44, _, X, Y, Z, a, b, y = qs[accepts[-1]]
c12 = -(c44 * 2.0 / anisotropic - c11)

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasis.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                 [c12, c11, c12, 0, 0, 0],
                 [c12, c12, c11, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c44, 0],
                 [0, 0, 0, 0, 0, c44]])

C, dCda, dCdb, dCdy, K = polybasis.buildRot(C, a, b, y)
K, M = polybasis.buildKM(C, dp, pv, density)
eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data) - 1))

for e1, dat in zip(eigs, data):
    print "{0:0.3f} {1:0.3f}".format(e1, dat)