#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

import itertools
import time
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

#%%
q1 = [0.703295, -0.111453, -0.101898, -0.69467299999999998]#0.988, 0.0, 0.006, -0.152]
#q = q / numpy.linalg.norm(q)

#q1 = quaternion.Quaternion(q)

q2 = [0.70376400000000006, -0.69423599999999996, -0.10509, 0.10820899999999999]#-0.425, -0.425, 0.565, 0.565]
#q = q / numpy.linalg.norm(q)

#q2 = quaternion.Quaternion(q)

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

    return misot

# to move q2 back to q1, use all the i, j, k rotations.... Not just miso
print miso((q1, q2))
