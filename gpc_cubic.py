#%%

import numpy
import time
import scipy
import sympy
import os
import sys
import matplotlib.pyplot as plt
sys.path.append('/home/bbales2/gpc')
os.chdir('/home/bbales2/modal')

import gpc
import rus
reload(rus)
#%%
#656 samples
#%%

# basis polynomials are x^n * y^m * z^l where n + m + l <= N
N = 12

## Dimensions for TF-2
X = 0.007753
Y = 0.009057
Z = 0.013199

#Sample density
density = 4401.695921 #Ti-64-TF2

std = 0.5

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

import polybasisqu

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
#%%

func(2.0, 1.0, 1.0)

#%%

reload(gpc)

func2 = lambda c11, c44 : func(c11, 1.0, c44)

minc11 = 1.5#1.6
maxc11 = 2.0#1.8
minc44 = 0.4#0.448
maxc44 = 0.6#0.452

hd = gpc.GPC(5, func2, [('u', (minc11, maxc11), 5),
                        ('u', (minc44, maxc44), 5)])

#%%
for r in range(5, len(data)):
    def L(c11, c44):
        logp = 0.0

        u = hd.approx(c11, c44)
        for i in range(0, r):#len(u)):
            logp += -0.5 * (data[i] - u[i])**2 / std**2 - numpy.log(std) - 0.5 * numpy.log(2.0) - 0.5 * numpy.log(numpy.pi)

        return numpy.exp(logp)

    denominator = hd.measure(L)

    minc11 = 1.6
    maxc11 = 1.8
    minc44 = 0.445
    maxc44 = 0.455

    post = []
    c11s = numpy.linspace(minc11, maxc11, 70)
    c44s = numpy.linspace(minc44, maxc44, 70)
    post = numpy.zeros((len(c11s), len(c44s)))
    for i, z1 in enumerate(c11s):
        for j, z2 in enumerate(c44s):
            post[i, j] = L(z1, z2) * hd.prior(z1, z2)

    post /= sum(sorted(denominator.flatten()))
            # / denominator#
    #for z in c11s:
    #    post.append(L(z) * hd.prior(z) / denominator)

    plt.imshow(post, interpolation = 'NONE', extent = [minc44, maxc44, maxc11, minc11], aspect = (maxc44 - minc44) / (maxc11 - minc11))
    #plt.colorbar()
    plt.xlabel('c44', fontsize = 36)
    plt.ylabel('c11', fontsize = 36)
    #plt.plot(c11s, post)
    #plt.xlabel('c11')
    plt.title('Posterior, {0} datapoints'.format(r), fontsize = 36)
    plt.gcf().set_size_inches((12, 8))
    plt.savefig("/home/bbales2/Documents/group_meeting/jan12/data/{0}.png".format(r), bbox_inches = 'tight', pad_inches = 0.0)
    plt.show()
#%%
