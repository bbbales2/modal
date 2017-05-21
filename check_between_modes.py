#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion
from rotations import inv_rotations

import itertools
import time
import pandas
import random
import matplotlib.pyplot as plt
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

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
                    qasb2 = qb * qa.conjugate()

                    t1 = qasb1.wxyz / numpy.linalg.norm(qasb1.wxyz)
                    t2 = qasb2.wxyz / numpy.linalg.norm(qasb2.wxyz)

                    a1 = 2 * numpy.arccos(t1[0]) * 180 / numpy.pi
                    a2 = 2 * numpy.arccos(t2[0]) * 180 / numpy.pi

                    if a1 < misot:
                        misot = a1
                        misoa = qasb1

                    if a2 < misot:
                        misot = a2
                        misoa = qasb2

    return misot

#%%
files = ['/home/bbales2/cmdstan-rus/examples/cu/cmsx4.20modes.1.csv',
         '/home/bbales2/cmdstan-rus/examples/cu/cmsx4.20modes.2.csv',
         '/home/bbales2/cmdstan-rus/examples/cu/cmsx4.20modes.3.csv',
         '/home/bbales2/cmdstan-rus/examples/cu/cmsx4.20modes.4.csv']

def getDist(qus1, qus2):
    angles = []
    for i, (q1, q2) in enumerate(zip(qus1, qus2)):
        angles.append(miso((q1, q2)))
        #print i
    return angles

qus_ = []
angles_ = []
for f in files:
    df = pandas.read_csv(f, comment = '#')

    qus = []
    for i in range(1000, 2000):
        qus.append([df.iloc[i]['q.1'], df.iloc[i]['q.2'], df.iloc[i]['q.3'], df.iloc[i]['q.4']])#inv_rotations.cu2qu([df.iloc[i]['cu.1'], df.iloc[i]['cu.2'], df.iloc[i]['cu.3']])

    random.shuffle(qus)

    angles = getDist(qus[::2], qus[1::2])

    plt.hist(angles, bins = 30, alpha = 0.33)

    qus_.append(qus)
    angles_.append(angles)
plt.show()
#%%
random.shuffle(qus_)
angles = getDist(qus_[::2], qus_[1::2])
plt.hist(angles, bins = 30)
plt.show()

#%%
for i in range(4):
    for j in range(i + 1, 4):
        qus = qus_[i] + qus_[j]

        random.shuffle(qus)

        angles = getDist(qus[::2], qus[1::2])
        plt.hist(angles)
        plt.title('i = {0}, j = {1}'.format(i, j))
        plt.show()
#%%
