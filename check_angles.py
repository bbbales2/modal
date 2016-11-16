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

def get_misorientations(g1, g2, n):
    angles = []

    idxs = range(0, len(g1))
    numpy.random.shuffle(idxs)
    idxs1 = idxs[0 : n]

    idxs = range(0, len(g2))
    numpy.random.shuffle(idxs)
    idxs2 = idxs[0 : n]

    for t, (ii, jj) in enumerate(zip(idxs1, idxs2)):
        q1 = quaternion.Quaternion(g1[ii])
        q2 = quaternion.Quaternion(g2[jj])

        deltaQ = q1 * q2.conjugate()

        cubicSym = symmetry.Symmetry.Cubic.quOperators()
        orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()
        miso = deltaQ
        for i in range(len(cubicSym)):
            qa = cubicSym[i] * q1
            for j in range(len(orthoSym)):
                qas = qa * orthoSym[j]
                for k in range(len(cubicSym)):
                    qasb = qas * q2.conjugate() * cubicSym[k]
                    miso = max([miso, qasb, qasb.conjugate()])

        print t / float(n)

        #print miso[0]

        angle = 2 * numpy.arccos(miso.wxyz[0]) * 180 / numpy.pi

        angle = 0 if numpy.isnan(angle) else angle

        angles.append(angle)

    # to move q2 back to q1, use all the i, j, k rotations.... Not just miso
    return angles

angles11 = get_misorientations(g1, g1, 250)
print '----'
angles12 = get_misorientations(g1, g2, 250)
print '----'
angles22 = get_misorientations(g2, g2, 250)
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