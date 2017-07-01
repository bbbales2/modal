#%%

import os
import pickle
import numpy
import scipy

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/modal')

from rotations import inv_rotations
from rotations import symmetry
from rotations import quaternion

with open('paper/cmsx4/hmc_30_noprior.pkl') as f:
    hmc = pickle.load(f)

samples = dict(zip(*hmc.format_samples()))

cubicSym = symmetry.Symmetry.Cubic.quOperators()
orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()

def adj(q):
    if q.wxyz[0] < 0.0:
        q.wxyz[0] *= -1
        q.wxyz[1] *= -1
        q.wxyz[2] *= -1
        q.wxyz[3] *= -1

    return q

def miso((q1, q2)):
    q1 = quaternion.Quaternion(numpy.array(q1) / numpy.linalg.norm(q1)).conjugate()
    q2 = quaternion.Quaternion(numpy.array(q2) / numpy.linalg.norm(q2)).conjugate()
    misot = 180.0
    misoa = None

    for i in range(len(cubicSym)):
        for ii in range(len(orthoSym)):
            qa = orthoSym[ii] * q1 * cubicSym[i]

            for j in range(len(cubicSym)):
                #for jj in range(len(orthoSym)):
                    qb = q2 * cubicSym[j]

                    qasb1 = qa.conjugate() * qb

                    t1 = qasb1.wxyz / numpy.linalg.norm(qasb1.wxyz)

                    a1 = 2 * numpy.arccos(t1[0]) * 180 / numpy.pi

                    if a1 < misot:
                        misot = a1
                        misoa = qasb1

    return misot, adj(misoa.conjugate())
#%%
print miso(([0.7018118488034941, 0.6943707989738772, 0.12351421786766208, 0.10026744434730868], [0.70181185,  0.69437080,  0.12351422,  0.10026744]))
print miso(([0.5671971160114844, -0.5822823918423391, 0.0, -0.5824385355886229], [0.70181185,  0.69437080,  0.12351422,  0.10026744]))
print miso(([0.6994423294760791, 0.6964960229650607, 0.12790702822920946, 0.09650652755384859], [0.70181185,  0.69437080,  0.12351422,  0.10026744]))
#%%
#cu1, cu2, cu3 = 0, 0, 0
#angle = 0
data = [["chain", "c11", "std", "c44", "a", "cu1", "cu2", "cu3", "w", "x", "y", "z", "angle"]]
for i in range(1, 5):
    with open("paper/cmsx4/only_one/{0}".format(i)) as f:
        for line in f:
            line = line.split(",")
            c11, std, c44, a, w, x, y, z = [float(line[j]) for j in range(1, 16, 2)]
            angle, q = miso(([0.70181185,  0.69437080,  0.12351422,  0.10026744], [w, x, y, z]))
            cu1, cu2, cu3 = inv_rotations.qu2cu(q)
            data.append([i, c11, std, c44, a, cu1, cu2, cu3, w, x, y, z, angle])

with open("paper/cmsx4/hmc_30_noprior_data_warmup.csv", "w") as f:
    for line in data:
        f.write(",".join(str(v) for v in line) + "\n")

#%%
data = [["chain", "c11", "std", "c44", "a", "cu1", "cu2", "cu3", "w", "x", "y", "z", "angle"]]
for i in range(1, 5):
    with open("/home/bbales2/chuckwalla/modal/paper/cmsx4/1k/chain{0}.txt".format(i)) as f:
        for line in f:
            line = line.split(",")
            c11, std, c44, a, w, x, y, z = [float(line[j]) for j in range(1, 16, 2)]
            angle, q = miso(([0.70181185,  0.69437080,  0.12351422,  0.10026744], [w, x, y, z]))
            cu1, cu2, cu3 = inv_rotations.qu2cu(q)
            data.append([i, c11, std, c44, a, cu1, cu2, cu3, w, x, y, z, angle])

with open("paper/cmsx4/hmc_30_noprior_data.csv", "w") as f:
    for line in data:
        f.write(",".join(str(v) for v in line) + "\n")
#%%
print inv_rotations.qu2cu([0.70181185,  0.69437080,  0.12351422,  0.10026744])

data = []
for i in range(1, 20):
    a, q = miso(([0.70181185,  0.69437080,  0.12351422,  0.10026744], [samples['w_0'][-i], samples['x_0'][-i], samples['y_0'][-i], samples['z_0'][-i]]))
    print a
    data.append([a] + inv_rotations.qu2cu(q.wxyz))

numpy.savetxt("paper/cmsx4/hmc_30_noprior_data.csv", data, delimiter = ",")
#%%
#
# Generate posterior predictive samples
#
#%%
N = 12

## Dimensions for TF-2
X = 0.011959
Y = 0.019976
Z = 0.013953

#Sample density
density = 8701.0#4401.695921

def func(c11, a, c44, w, x, y, z):
    c12 = -(c44 * 2.0 / a - c11)

    q = numpy.array([w, x, y, z])
    q /= numpy.linalg.norm(q)
    w, x, y, z = q.flatten()

    C = numpy.array([[c11, c12, c12, 0, 0, 0],
                     [c12, c11, c12, 0, 0, 0],
                     [c12, c12, c11, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, c44]])

    dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)

    C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

    K, M = polybasisqu.buildKM(C, dp, pv, density)
    eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + 30 - 1))

    return numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

#%%
import polybasisqu
data = []

for i in range(1, 5):
    with open("/home/bbales2/chuckwalla/modal/paper/cmsx4/1k/chain{0}.txt".format(i)) as f:
        for line in f:
            line = line.split(",")
            c11, std, c44, a, w, x, y, z = [float(line[j]) for j in range(1, 16, 2)]
            data.append(func(c11, a, c44, w, x, y, z) + numpy.random.randn(30) * std)

#%%
data2 = numpy.array(data)

std = numpy.std(data2, axis = 0)
mm = numpy.mean(data2, axis = 0)
upp = numpy.percentile(data2, 97.5, axis = 0)
loo = numpy.percentile(data2, 2.5, axis = 0)

meas = [71.25925, 75.75875, 86.478, 89.947375, 111.150125,
        112.164125, 120.172125, 127.810375, 128.6755, 130.739875,
        141.70025, 144.50375, 149.40075, 154.35075, 156.782125,
        157.554625, 161.0875, 165.10325, 169.7615, 173.44925,
        174.11675, 174.90625, 181.11975, 182.4585, 183.98625,
        192.68125, 193.43575, 198.793625, 201.901625, 205.01475]

for i, (m, dat, st, lo, up) in enumerate(zip(mm, meas, std, loo, upp)):
    o = '\\rowcolor{lightgrayX50}' if (dat < m - 2.0 * st or dat > m + 2.0 * st) else ''
    print "{4} {0} & ${1:0.2f}$ & ${2:0.3f} \pm {3:0.2f}$ \\\\".format(i + 1, dat, m, st, o)