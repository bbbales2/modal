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
N = 14

density = 8700.0  #4401.695921#

# Dimensions -- watch the scaling
X = .011  #0.007753#
Y = .013  #0.009057#
Z = .019  #0.013199#

c11 = 2.6
anisotropic = 2.8421
c44 = 1.35
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
data = numpy.array([
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

#%%

c12 = -(c44 * 2.0 / anisotropic - c11)

dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

C = numpy.array([[c11, c12, c12, 0, 0, 0],
                 [c12, c11, c12, 0, 0, 0],
                 [c12, c12, c11, 0, 0, 0],
                 [0, 0, 0, c44, 0, 0],
                 [0, 0, 0, 0, c44, 0],
                 [0, 0, 0, 0, 0, c44]])

w, x, y, z = 0.594755820, -0.202874980, 0.640151553, 0.441942582
#w, x, y, z = 1.0, 0.0, 0.0, 0.0
#w, x, y, z = [0.87095, 0.17028, 0.03090, 0.45989]
#w, x, y, z = [0.93894, -0.09845, -0.14279, -0.29717]

C, dCdw, dCdx, dCdy, dCdz, Kt = polybasisqu.buildRot(C, w, x, y, z)
K, M = polybasisqu.buildKM(C, dp, pv, density)
eigs2, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 30 - 1))

freqs = numpy.sqrt(eigs2 * 1e11) / (numpy.pi * 2000)

print "computed, accepted"
for e1, dat in zip(freqs, data):
    print "{0:0.5f} {1:0.3f}".format(e1, dat)

#freqs + 0.25 * numpy.random.randn(len(freqs))
#%%
dCdc11 = numpy.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]], dtype = 'float64')

dCdc11 = Kt.dot(dCdc11).dot(Kt.T)

dCdc12 = numpy.array([[0, 1, 1, 0, 0, 0],
                 [1, 0, 1, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]], dtype = 'float64')

dCdc12 = Kt.dot(dCdc12).dot(Kt.T)

dCdc44 = numpy.array([[0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0, 1]], dtype = 'float64')

dCdc44 = Kt.dot(dCdc44).dot(Kt.T)

if True:
    dKdw, _ = polybasisqu.buildKM(dCdw, dp, pv, density)
    dKdx, _ = polybasisqu.buildKM(dCdx, dp, pv, density)
    dKdy, _ = polybasisqu.buildKM(dCdy, dp, pv, density)
    dKdz, _ = polybasisqu.buildKM(dCdz, dp, pv, density)

    dKdc11, _ = polybasisqu.buildKM(dCdc11, dp, pv, density)
    dKdc12, _ = polybasisqu.buildKM(dCdc12, dp, pv, density)
    dKdc44, _ = polybasisqu.buildKM(dCdc44, dp, pv, density)

    dldw = numpy.array([evecs[:, i].T.dot(dKdw.dot(evecs[:, i])) for i in range(evecs.shape[1])])
    dldx = numpy.array([evecs[:, i].T.dot(dKdx.dot(evecs[:, i])) for i in range(evecs.shape[1])])
    dldy = numpy.array([evecs[:, i].T.dot(dKdy.dot(evecs[:, i])) for i in range(evecs.shape[1])])
    dldz = numpy.array([evecs[:, i].T.dot(dKdz.dot(evecs[:, i])) for i in range(evecs.shape[1])])
    dldc11 = numpy.array([evecs[:, i].T.dot(dKdc11.dot(evecs[:, i])) for i in range(evecs.shape[1])])
    dldc12 = numpy.array([evecs[:, i].T.dot(dKdc12.dot(evecs[:, i])) for i in range(evecs.shape[1])])
    dldc44 = numpy.array([evecs[:, i].T.dot(dKdc44.dot(evecs[:, i])) for i in range(evecs.shape[1])])
#%%
for a, b, c in zip(dldc11, dldc12, dldc44):
    print a, b, c
#%%
for f1, f2, f3 in zip(freqs1, freqs2, freqs3[:30]):
    print ", ".join(["{0:.2f}".format(a) for a in [f1, f2, f3]])
#%%

print "minimum (y = -0.015), y = 0.0, measured, error vs. y = -0.015, error vs. y = 0.0"
for e1, e2, dat in zip(eigs, eigs2, data):
    print "{0:0.3f} {1:0.3f} {2:0.3f} {3:0.3f} {4:0.3f}".format(e1, e2, dat, numpy.abs(e1 - dat), numpy.abs(e2 - dat))
