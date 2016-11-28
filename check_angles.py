#%%

import os
import sklearn.mixture
import numpy
import matplotlib.pyplot as plt

os.chdir("/home/bbales2/modal")

qs = numpy.loadtxt("/home/bbales2/modal/paper/cmsx4/qs.csv", delimiter = ",", skiprows = 1)

#%%

gmm = sklearn.mixture.GMM(n_components = 2, min_covar = 1e-9)

data = qs[-2000:, 4:]

C = gmm.fit_predict(data)

print gmm.weights_
print gmm.means_
print gmm.covars_

plt.hist(data[:, 2])
plt.show()

plt.hist(gmm.sample(2000))
plt.show()
#%%
print numpy.mean(qs[-2000:][C == 0], axis = 0)
print numpy.mean(qs[-2000:][C == 1], axis = 0)
#%%
g0 = qs[-2000:, 4:]
g1 = qs[-2000:, 4:][C == 0]
g2 = qs[-2000:, 4:][C == 1]
#%%

from rotations import symmetry
from rotations import quaternion

cubicSym = symmetry.Symmetry.Cubic.quOperators()

def get_misorientations(g1, g2, n):
    angles = []

    idxs = range(0, len(g1))
    numpy.random.shuffle(idxs)
    idxs1 = idxs[0 : n]

    idxs = range(0, len(g2))
    numpy.random.shuffle(idxs)
    idxs2 = idxs[0 : n]

    for t, (ii, jj) in enumerate(zip(idxs1, idxs2)):
        gg1 = g1[ii]
        gg2 = g2[jj]

        q1 = quaternion.Quaternion(gg1 / numpy.linalg.norm(gg1))
        q2 = quaternion.Quaternion(gg2 / numpy.linalg.norm(gg2))

        miso = 180.0
        misoa = None

        def adj(q):
            if q.wxyz[0] < 0.0:
                q.wxyz[0] *= -1
                q.wxyz[1] *= -1
                q.wxyz[2] *= -1
                q.wxyz[3] *= -1

            return q

        for i in range(len(cubicSym)):
            qa = adj(cubicSym[i] * q1)

            for k in range(len(cubicSym)):
                qb = adj(cubicSym[k] * q2)

                qasb1 = adj(qa * qb.conjugate())
                qasb2 = adj(qb * qa.conjugate())

                t1 = qasb1.wxyz / numpy.linalg.norm(qasb1.wxyz)
                t2 = qasb2.wxyz / numpy.linalg.norm(qasb2.wxyz)
                #print qasb1
                #print qasb2
                #1/0
                a1 = 2 * numpy.arccos(t1[0]) * 180 / numpy.pi
                a2 = 2 * numpy.arccos(t2[0]) * 180 / numpy.pi

                if a1 < miso:
                    #print qasb1
                    #print qasb2
                    miso = a1
                    misoa = qasb1

                if a2 < miso:
                    #print qasb1
                    #print qasb2
                    miso = a2
                    #print 'hi2', a2, miso
                    misoa = qasb2
                    #print 'hi', a1, miso

                #print 'miso', q1, q2, miso

        print t / float(n)

        #if miso > 50.0:
        #    print q1, q2, miso, t, ii, jj
        #    1/0

        angles.append(miso)

    # to move q2 back to q1, use all the i, j, k rotations.... Not just miso
    return angles
#%%
angles11 = get_misorientations([g1[14]], [g1[14]], 1000)
#%%
angles11 = get_misorientations(g1, g1, 1000)
print '----'
angles12 = get_misorientations(g1, g2, 1000)
print '----'
angles22 = get_misorientations(g2, g2, 1000)
#%%
symmetry.Symmetry.Orthorhombic.quInFZ(quaternion.Quaternion(g1[0]))
#%%
print numpy.mean(angles11)
print numpy.std(angles11)

plt.hist(angles11)
plt.xlabel('Misorientation angle in degrees')
plt.title('Misorientation of group 1 with respect to group 1')
plt.show()

print numpy.mean(angles12)
print numpy.std(angles12)

plt.hist(angles12)
plt.xlabel('Misorientation angle in degrees')
plt.title('Misorientation of group 1 with respect to group 2')
plt.show()

print numpy.mean(angles22)
print numpy.std(angles22)

plt.hist(angles22)
plt.xlabel('Misorientation angle in degrees')
plt.title('Misorientation of group 2 with respect to group 2')
plt.show()
#%%
print(miso)
print "Difference in angle: {0} degrees".format(2 * numpy.arccos(miso.wxyz[0]) * 180 / numpy.pi)
#%%
ws = qs[-2000:, 4]
xs = qs[-2000:, 5]
ys = qs[-2000:, 6]
zs = qs[-2000:, 7]

Ws = []
for w, x, y, z in zip(ws, xs, ys, zs):
    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    Ws.append(Q.T.dot([1.0, 0.0, 0.0]))

plt.plot(Ws)
plt.legend(['x-components', 'y-components', 'z-components'])
plt.show()
