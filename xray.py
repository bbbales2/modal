#%%

import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

from rotations import inv_rotations

inv_rotations.convention = inv_rotations.Convention.passive
q3 = inv_rotations.eu2qu(numpy.array([180.9, 89.3, 338.6]) * numpy.pi / 180.0)

phi = -95.64
chi = 18.266
q1 = numpy.array([numpy.cos(chi * numpy.pi / 360.0), 0.0, -numpy.sin(chi * numpy.pi / 360.0), 0.0])
q2 = numpy.array([numpy.cos(phi * numpy.pi / 360.0), 0.0, 0.0, -numpy.sin(phi * numpy.pi / 360.0)])

q1 = quaternion.Quaternion(q1 / numpy.linalg.norm(q1))
q2 = quaternion.Quaternion(q2 / numpy.linalg.norm(q2))
q100 = q1 * q2

phi = -90.688
chi = 63.266
q1 = numpy.array([numpy.cos(chi * numpy.pi / 360.0), 0.0, -numpy.sin(chi * numpy.pi / 360.0), 0.0])
q2 = numpy.array([numpy.cos(phi * numpy.pi / 360.0), 0.0, 0.0, -numpy.sin(phi * numpy.pi / 360.0)])
q3 = numpy.array([numpy.cos(45.0 * numpy.pi / 360.0), 0.0, 0.0, -numpy.sin(45.0 * numpy.pi / 360.0)])

q1 = quaternion.Quaternion(q1 / numpy.linalg.norm(q1))
q2 = quaternion.Quaternion(q2 / numpy.linalg.norm(q2))
q3 = quaternion.Quaternion(q3 / numpy.linalg.norm(q3))
q110 = q1 * q2 * q3

print misoo(q100, q110)
#print q3
#%%
import itertools

def misoo(q1, q2):
    cubicSym = symmetry.Symmetry.Cubic.quOperators()
    orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()

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
        for ii in range(len(orthoSym)):
            qa = orthoSym[ii] * q1 * cubicSym[i]

            for j in range(len(cubicSym)):
                for jj in range(len(orthoSym)):
                    qb = orthoSym[jj] * q2 * cubicSym[j]

                    qasb1 = qa.conjugate() * qb

                    t1 = qasb1.wxyz / numpy.linalg.norm(qasb1.wxyz)

                    a1 = 2 * numpy.arccos(t1[0]) * 180 / numpy.pi

                    if a1 < miso:
                        miso = a1
                        misoa = qasb1

    return miso, misoa
