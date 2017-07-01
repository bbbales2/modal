#%%

import os
import pickle
import pyximport
import numpy
import scipy

os.chdir('/home/bbales2/modal')

from rotations import inv_rotations
from rotations import symmetry
from rotations import quaternion

#
# Generate posterior predictive samples
#
#%%
N = 12

## Dimensions for TF-2
X = 0.007753
Y = 0.009057
Z = 0.013199

#Sample density
density = 4401.695921

def func(c11, a, c44):
    c12 = -(c44 * 2.0 / a - c11)

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 30 - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

#%%
import polybasisqu
data = []

for i in range(1, 2):
    with open("/home/bbales2/chuckwalla/modal/paper/ti/paper_initial_conditions/chain{0}.txt".format(i)) as f:
        for line in f:
            line = line.strip().split(",")
            c11, std, c44, a = [float(line[j]) for j in range(1, 8, 2)]
            data.append(func(c11, a, c44) + numpy.random.randn(30) * std)

#%%
data2 = numpy.array(data)

std = numpy.std(data2, axis = 0)
mm = numpy.mean(data2, axis = 0)
upp = numpy.percentile(data2, 97.5, axis = 0)
loo = numpy.percentile(data2, 2.5, axis = 0)

meas = [109.076, 136.503, 144.899, 184.926, 188.476,
195.562, 199.246, 208.460, 231.220, 232.630,
239.057, 241.684, 242.159, 249.891, 266.285,
272.672, 285.217, 285.670, 288.796, 296.976,
301.101, 303.024, 305.115, 305.827, 306.939,
310.428, 318.000, 319.457, 322.249, 323.464]

for i, (m, dat, st, lo, up) in enumerate(zip(mm, meas, std, loo, upp)):
    o = '\\rowcolor{lightgrayX50}' if (dat < lo or dat > up) else ''
    print "{4} {0} & ${1:0.2f}$ & ${2:0.3f} \pm {3:0.2f}$ \\\\".format(i + 1, dat, m, st, o)