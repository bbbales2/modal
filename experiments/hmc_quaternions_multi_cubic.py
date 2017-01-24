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

import contextlib

# Stolen from http://stackoverflow.com/a/2891805/3769360
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield
    numpy.set_printoptions(**original)

#from rotations import symmetry
#from rotations import quaternion
#from rotations import inv_rotations

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 10

densities = [8700.0, 8700.0]  #4401.695921#

# Dimensions -- watch the scaling
Xs = [.011959, .011959]  #0.007753#
Ys = [.013953, .013953]  #0.009057#
Zs = [.019976, .019976]  #0.013199#

S = len(Xs)

c11 = 2.5
anisotropic = 2.8
c44 = 1.3
c12 = -(c44 * 2.0 / anisotropic - c11)

# Standard deviation around each mode prediction
std = 1.0

# Rotations
ws = [1.0] * S
xs = [0.0] * S
ys = [0.0] * S
zs = [0.0] * S

# These are the sampled modes in khz

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

data = []
for s in range(S):
    data.append(freqs + numpy.random.randn(len(freqs)) * 1.0)
#%%
qs = []
rs = []
logps = []
accepts = []

current_q = numpy.array([c11, anisotropic, c44, std])
current_r = numpy.array([ws, xs, ys, zs]).transpose()

#%%
# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50

# start epsilon at .0001 and try larger values like .0005 after running for a while
epsilon = 0.00125

# Set this to true to debug the L and eps values
debug = False#False#False#True#

#%%
dps = []
pvs = []

for X, Y, Z in zip(Xs, Ys, Zs):
    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    dps.append(dp)
    pvs.append(pv)

# This is for running the HMC
def UgradU(q, r):
    c11, anisotropic, c44, std = q
    c12 = -(c44 * 2.0 / anisotropic - c11)
    #print q#X, Y, Z

    #tmp = time.time()
    #print "Basis build: ", time.time() - tmp

    #tmp = time.time()

    total_logp = 0.0
    total_dlogpdq = numpy.zeros(q.shape)
    total_dlogpdr = numpy.zeros(r.shape)

    for s, density in zip(range(S), densities):
        w, x, y, z = r[s]

        C = numpy.array([[c11, c12, c12, 0, 0, 0],
                         [c12, c11, c12, 0, 0, 0],
                         [c12, c12, c11, 0, 0, 0],
                         [0, 0, 0, c44, 0, 0],
                         [0, 0, 0, 0, c44, 0],
                         [0, 0, 0, 0, 0, c44]])

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

        dKdw, _ = polybasisqu.buildKM(dCdw, dp, pv, density)
        dKdx, _ = polybasisqu.buildKM(dCdx, dp, pv, density)
        dKdy, _ = polybasisqu.buildKM(dCdy, dp, pv, density)
        dKdz, _ = polybasisqu.buildKM(dCdz, dp, pv, density)

        dKdc11, _ = polybasisqu.buildKM(dCdc11, dp, pv, density)
        dKdc12, _ = polybasisqu.buildKM(dCdc12, dp, pv, density)
        dKdc44, _ = polybasisqu.buildKM(dCdc44, dp, pv, density)

        K, M = polybasisqu.buildKM(C, dp, pv, density)

        eigst, evecst = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data[s]) - 1))
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

        freqst = numpy.sqrt(eigst * 1e11) / (numpy.pi * 2000)#(freqs * numpy.pi * 2000) ** 2 / 1e11

        dfreqsdl = 0.5e11 / (numpy.sqrt(eigst * 1e11) * numpy.pi * 2000)

        dlpdfreqs = (data[s] - freqst) / std ** 2
        dlpdstd = sum((-std ** 2 + (freqst - data[s]) **2) / std ** 3)

        dlpdl = dfreqsdl * dlpdfreqs

        dlpdw = dlpdl.dot(dldw)
        dlpdx = dlpdl.dot(dldx)
        dlpdy = dlpdl.dot(dldy)
        dlpdz = dlpdl.dot(dldz)
        dlpdc11 = dlpdl.dot(dldc11)
        dlpdc12 = dlpdl.dot(dldc12)
        dlpdc44 = dlpdl.dot(dldc44)

        logp = sum(0.5 * (-((freqst - data[s]) **2 / std**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

        dlpdc12tf = dlpdc12 * 2.0 * c44 / (anisotropic**2)

        total_logp += logp
        #print numpy.array([dlpdc11 + dlpdc12, dlpdc12tf, dlpdc44 + dlpdc12 * -2 / anisotropic, dlpdstd])
        #print total_dlogpdq
        total_dlogpdq += numpy.array([dlpdc11 + dlpdc12, dlpdc12tf, dlpdc44 + dlpdc12 * -2 / anisotropic, dlpdstd])
        #print numpy.array([dlpdw, dlpdx, dlpdy, dlpdz])
        total_dlogpdr[s, :] = numpy.array([dlpdw, dlpdx, dlpdy, dlpdz])

    return -total_logp, -total_dlogpdq, -total_dlogpdr

while True:
    q = current_q.copy()
    r = current_r.copy()

    p = numpy.random.randn(len(q)) # independent standard normal variates
    pr = numpy.random.randn(*r.shape)

    for s in range(S):
        pr[s] -= numpy.outer(r[s], r[s]).dot(pr[s])

    current_p = p.copy()
    current_pr = pr.copy()

    #print current_p, current_q
    # Make a half step for momentum at the beginning
    U, gradUp, gradUr = UgradU(q, r)
    p = p - epsilon * gradUp / 2
    pr = pr - epsilon * gradUr / 2

    for s in range(S):
        pr[s] -= numpy.outer(r[s], r[s]).dot(pr[s])

    # Alternate full steps for position and momentum
    for i in range(L):
        # Make a full step for the position
        q = q + epsilon * p

        for s in range(S):
            alpha = numpy.linalg.norm(pr[s])

            m1 = numpy.array([[1.0, 0.0],
                              [0.0, 1 / alpha]])

            m2 = numpy.array([[numpy.cos(alpha * epsilon), -numpy.sin(alpha * epsilon)],
                              [numpy.sin(alpha * epsilon), numpy.cos(alpha * epsilon)]])

            m3 = numpy.array([[1.0, 0.0],
                              [0.0, alpha]])

            xv = numpy.array([r[s], pr[s]]).T.dot(m1.dot(m2.dot(m3)))

            r[s] = xv[:, 0]
            pr[s] = xv[:, 1]

            r[s] /= numpy.linalg.norm(r[s])

        # Make a full step for the momentum, except at end of trajectory
        if i != L - 1:
            U, gradUp, gradUr = UgradU(q, r)
            p = p - epsilon * gradUp
            pr = pr - epsilon * gradUr

            for s in range(S):
                pr[s] -= numpy.outer(r[s], r[s]).dot(pr[s])

        if debug:
            with printoptions(precision = 5):
                print "H (constant or decreasing): ", U + sum(p ** 2) / 2 + (pr ** 2).sum() / 2, U, sum(p ** 2) / 2, (pr ** 2).sum() / 2
                print "New q: ", q
                print r

    U, gradUp, gradUr = UgradU(q, r)
    # Make a half step for momentum at the end.
    p = p - epsilon * gradUp / 2
    pr = pr - epsilon * gradUr / 2

    for s in range(S):
        pr[s] -= numpy.outer(r[s], r[s]).dot(pr[s])

    # Negate momentum at end of trajectory to make the proposal symmetric
    p = -p
    pr = -pr
    # Evaluate potential and kinetic energies at start and end of trajectory
    #print current_p, current_q
    UC, _, _ = UgradU(current_q, current_r)
    current_U = UC
    current_K = sum(current_p ** 2) / 2 + (current_pr ** 2).sum() / 2
    proposed_U = U
    proposed_K = sum(p ** 2) / 2 + (pr ** 2).sum() / 2

    # Accept or reject the state at end of trajectory, returning either
    # the position at the end of the trajectory or the initial position
    dQ = current_U - proposed_U + current_K - proposed_K

    logps.append(UC)

    with printoptions(precision=3):
        if numpy.random.rand() < min(1.0, numpy.exp(dQ)):
            current_q = q # accept
            current_r = r

            accepts.append(len(qs) - 1)

            print "Accepted ({0} accepts so far): {1}".format(len(accepts), current_q)
            print current_r
        else:
            print "Rejected: ", current_q
            print current_r

    rs.append(current_r.copy())
    qs.append(current_q.copy())
    print "Norm of rotation vector: ", numpy.linalg.norm(q[-4:])
    print "Energy change ({0} samples, {1} accepts): ".format(len(qs), len(accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K
    print "Epsilon: ", epsilon
#%%
import sklearn.mixture

gmm = sklearn.mixture.GMM(2, min_covar = 1e-12)

gmm.fit(numpy.array(rs)[-1000:, 0])

print gmm.means_
print numpy.mean(qs[-1000:], axis = 0)
#%%

import matplotlib.pyplot as plt

s0 = numpy.array([2.48673373, 2.86907127, 1.31391957, 0.89182735, 0.68331817, 0.09616822, -0.71426889, 0.11549438])
s1 = numpy.array([2.48673373, 2.86907127, 1.31391957, 0.89182735, 0.71498083, 0.11774449, -0.68248917, 0.0940223])

ds = s1 - s0

for N in [8, 10, 12]:
    Us = []
    xs = numpy.linspace(-0.25, 1.25, 20)
    for i in range(len(xs)):
        t = s0 + ds * xs[i]

        print t

        t[4] = numpy.sqrt(1.0 - t[5]**2 - t[6]**2 - t[7]**2)

        U, gradU, gradUr = UgradU(t[:4], numpy.array([t[4:], t[4:]]))

        Us.append(U)
        print "{0}/{1}".format(i, len(xs))

    plt.plot(xs, Us)
    plt.title("N = {0}".format(N))
    plt.xlabel('y')
    plt.ylabel('-logp')
    plt.show()
#%%
f = open("/home/bbales2/modal/paper/cmsx4/qs2.csv", "w")
f.write("\n")
for q, r in zip(qs, rs):
    f.write(", ".join([str(a) for a in numpy.concatenate((q, r[0]))]) + "\n")
f.close()
#%%
numpy.savetxt("/home/bbales2/modal/paper/cmsx4/qs.csv", qs, delimiter = ",", comments = "", header = "c11, anisotropic, c44, std, ws, xs, ys, zs")
#%%
# Save samples (qs)
# First argument is filename

import os
import tempfile
import datetime

_, filename = tempfile.mkstemp(prefix = "data_{0}_".format(datetime.datetime.now().strftime("%Y-%m-%d")), suffix = ".txt", dir = os.getcwd())
header = "c11 anisotropic c44 std"
data = []
for s in range(S):
    header += ' w{0} x{0} y{0} z{0}'.format(s)

for q, r in zip(qs, rs):
    tmp = []

    tmp.extend(q)
    for s in range(S):
        tmp.extend(r[s])

    data.append(tmp)

numpy.savetxt(filename, data, header = header)
print filename, "saved"
#%%
# This is for plotting the trajectory of the samples through space
c11s, anisotropics, c44s, stds  = [numpy.array(a) for a in zip(*qs)]#
#, ws, xs, ys, zs
import matplotlib.pyplot as plt

for name, data1 in zip(['c11', 'anisotropics', 'c44', 'stds'],#, 'ws', 'xs', 'ys', 'zs'],
                      [c11s, anisotropics, c44s, stds]):#, ws, xs, ys, zs]):
    plt.plot(data1)
    plt.title('{0} u = {1:.3e}, std = {2:.3e}'.format(name, numpy.mean(data1), numpy.std(data1)))
    plt.show()

for s in range(S):
    ws, xs, ys, zs = zip(*numpy.array(rs)[:, s])

    plt.subplot(221)
    plt.plot(ws)
    plt.title('w u = {01:.3e}, std = {1:.3e}'.format(numpy.mean(data1), numpy.std(data1)))
    plt.subplot(222)
    plt.plot(xs)
    plt.title('x u = {0:.3e}, std = {1:.3e}'.format(numpy.mean(data1), numpy.std(data1)))
    plt.subplot(223)
    plt.plot(ys)
    plt.title('y u = {0:.3e}, std = {1:.3e}'.format(numpy.mean(data1), numpy.std(data1)))
    plt.subplot(224)
    plt.plot(zs)
    plt.title('z u = {0:.3e}, std = {1:.3e}'.format(numpy.mean(zs), numpy.std(zs)))
    plt.show()

#%%
# This is for plotting distributions of the parameters

c11s, anisotropics, c44s, stds  = [numpy.array(a)[-500:] for a in zip(*qs)]#
import matplotlib.pyplot as plt

for name, data1 in zip(['c11', 'anisotropics', 'c44', 'stds'],
                      [c11s, anisotropics, c44s, stds]):
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.4f}, $\sigma$ = {2:0.4f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()

for s in range(S):
    ws, xs, ys, zs = zip(*numpy.array(rs)[-500:, s])

    plt.subplot(221)
    seaborn.distplot(ws, kde = False, fit = scipy.stats.norm)
    plt.title('w u = {01:.3e}, std = {1:.3e}'.format(numpy.mean(data1), numpy.std(data1)))
    plt.subplot(222)
    seaborn.distplot(xs, kde = False, fit = scipy.stats.norm)
    plt.title('x u = {0:.3e}, std = {1:.3e}'.format(numpy.mean(data1), numpy.std(data1)))
    plt.subplot(223)
    seaborn.distplot(ys, kde = False, fit = scipy.stats.norm)
    plt.title('y u = {0:.3e}, std = {1:.3e}'.format(numpy.mean(data1), numpy.std(data1)))
    plt.subplot(224)
    seaborn.distplot(zs, kde = False, fit = scipy.stats.norm)
    plt.title('z u = {0:.3e}, std = {1:.3e}'.format(numpy.mean(data1), numpy.std(data1)))
    plt.show()#%%

#%%
# Forward problem

# This snippet is helpful to test the last accepted sample

c11, anisotropic, c44, std = current_q
c12 = -(c44 * 2.0 / anisotropic - c11)

for s, (w, x, y, z), X, Y, Z, density in zip(range(S), current_r, Xs, Ys, Zs, densities):
    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, w, x, y, z)
    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs2, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + len(data[s]) - 1))

    freqst = numpy.sqrt(eigs2 * 1e11) / (numpy.pi * 2000)#(freqs * numpy.pi * 2000) ** 2 / 1e11
    print numpy.mean(freqst - data[s])
    print numpy.std(freqst - data[s])

    print "computed, accepted"
    for e1, dat in zip(freqst, data[s]):
        print "{0:0.5f} {1:0.3f}".format(e1, dat)
#%%
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE

import sklearn.mixture

gmm = sklearn.mixture.GMM(2, min_covar = 1e-12)

gmm.fit(numpy.array(rs)[-1000:, 0])

print gmm.means_
print numpy.mean(qs[-1000:], axis = 0)
#%%

import matplotlib.pyplot as plt

s0 = numpy.array([2.48673373, 2.86907127, 1.31391957, 0.89182735, 0.68331817, 0.09616822, -0.71426889, 0.11549438])
s1 = numpy.array([2.48673373, 2.86907127, 1.31391957, 0.89182735, 0.71498083, 0.11774449, -0.68248917, 0.0940223])

ds = s1 - s0

for N in [8, 10, 12]:
    Us = []
    xs = numpy.linspace(-0.25, 1.25, 20)
    for i in range(len(xs)):
        t = s0 + ds * xs[i]

        print t

        t[4] = numpy.sqrt(1.0 - t[5]**2 - t[6]**2 - t[7]**2)

        U, gradU, gradUr = UgradU(t[:4], numpy.array([t[4:], t[4:]]))

        Us.append(U)
        print "{0}/{1}".format(i, len(xs))

    plt.plot(xs, Us)
    plt.title("N = {0}".format(N))
    plt.xlabel('y')
    plt.ylabel('-logp')
    plt.show()
#%%
f = open("/home/bbales2/modal/paper/cmsx4/qs2.csv", "w")
f.write("\n")
for q, r in zip(qs, rs):
    f.write(", ".join([str(a) for a in numpy.concatenate((q, r[0]))]) + "\n")
f.close()
#%%
numpy.savetxt("/home/bbales2/modal/paper/cmsx4/qs.csv", qs, delimiter = ",", comments = "", header = "c11, anisotropic, c44, std, ws, xs, ys, zs")

# This is for seeing the passive rotation applied to a reference x-axis (to see what things converged too)
Ws = []
ws, xs, ys, zs = zip(*numpy.array(rs)[:, 0])
for w, x, y, z in zip(ws, xs, ys, zs):
    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    Ws.append(Q.T.dot([1.0, 0.0, 0.0]))

plt.plot(Ws)
plt.legend(['x-components', 'y-components', 'z-components'])
plt.show()

#%%

print "minimum (y = -0.015), y = 0.0, measured, error vs. y = -0.015, error vs. y = 0.0"
for e1, e2, dat in zip(eigs, eigs2, data):
    print "{0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f}".format(e1, e2, dat, numpy.abs(e1 - dat), numpy.abs(e2 - dat))
