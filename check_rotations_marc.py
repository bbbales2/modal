#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

import itertools
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

#q = [0.98849515554696088, -0.0026722254086214156, 0.015326741589590076, 0.15045024979640861]
q = (0.981, 0.0, 0.0, -0.1908)
#q = [ 0.98805806, -0.00772966, -0.00900226, -0.15362448]#[0.98845476, -0.00572989, -0.0177229, -0.15036705]
q = [0.594756, -0.202874, 0.640152, 0.441943]
q = q / numpy.linalg.norm(q)

q1 = quaternion.Quaternion(q)

#q = [0.98849515554696088, -0.0026722254086214156, -0.015326741589590076, 0.15045024979640861]
q = (0.988, 0.0, -0.01, 0.152)
#q = [ 0.98805806, -0.00772966, -0.00900226, -0.15362448]#[0.98845476, -0.00572989, -0.0177229, -0.15036705]
q = [0.874, -0.170, -0.033, 0.455]#[0.928, -0.129, -0.315, -0.141]
q = q / numpy.linalg.norm(q)

q2 = quaternion.Quaternion(q)

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
    qa = adj(cubicSym[i] * q1)

    for j in range(len(cubicSym)):
        qb = adj(cubicSym[j] * q2)

        qasb1 = adj(qa * qb.conjugate())
        qasb2 = adj(qb * qa.conjugate())

        t1 = qasb1.wxyz / numpy.linalg.norm(qasb1.wxyz)
        t2 = qasb2.wxyz / numpy.linalg.norm(qasb2.wxyz)

        a1 = 2 * numpy.arccos(t1[0]) * 180 / numpy.pi
        a2 = 2 * numpy.arccos(t2[0]) * 180 / numpy.pi

                #a1 = 2 * numpy.arccos(qasb1.wxyz[0]) * 180 / numpy.pi
                #a2 = 2 * numpy.arccos(qasb2.wxyz[0]) * 180 / numpy.pi

        if a1 < miso:
            miso = a1
            misoa = qasb1
            print t1, t2

        if a2 < miso:
            miso = a2
            misoa = qasb2
            print 'b', t1, t2
            1/0
            #miso = min(miso, )
        #miso = min(miso, 2 * numpy.arccos(qasb2.wxyz[0]) * 180 / numpy.pi)
        #2 * numpy.arccos(qasb2.wxyz[0]) * 180 / numpy.pi

        #miso = max([miso, qasb1, qasb2])
# to move q2 back to q1, use all the i, j, k rotations.... Not just miso
print(miso)
print misoa
#%%
def getvec(miso):
    ang = numpy.arccos(miso.wxyz[0])
    tmp = numpy.sin(ang)
    return [miso.wxyz[1] / tmp, miso.wxyz[2] / tmp, miso.wxyz[3] / tmp]

#print "Difference in angle: {0} degrees".format(2 * numpy.arccos(miso.wxyz[0]) * 180 / numpy.pi)

    #%%
def get_misorientations(g1, g2, n):
    angles = []

    idxs = range(0, len(g1))
    #numpy.random.shuffle(idxs)
    idxs1 = idxs[0 : n]

    idxs = range(0, len(g2))
    #numpy.random.shuffle(idxs)
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

                a1 = 2 * numpy.arccos(t1.wxyz[0]) * 180 / numpy.pi
                a2 = 2 * numpy.arccos(t2.wxyz[0]) * 180 / numpy.pi

                if a1 < miso:
                    miso = a1
                    misoa = qasb1

                if a2 < miso:
                    miso = a2
                    print 'hi2', a2, miso
                    misoa = qasb2
                    print 'hi', a1, miso

                print 'miso', q1, q2, miso

        print t / float(n)

        if miso > 50.0:
            print q1, q2, miso, t, ii, jj
            1/0

        angles.append(miso)

    # to move q2 back to q1, use all the i, j, k rotations.... Not just miso
    return angles

get_misorientations([q1.wxyz], [q2.wxyz], 1)
#%%
from rotations import quaternion
from rotations import inv_rotations

#q1 = quaternion.Quaternion((0.981, 0.0, 0.0, -0.1908))
#q2 = quaternion.Quaternion((0.988, 0.0, -0.01, 0.152))

#q1 = quaternion.Quaternion([0.594756, -0.202874, 0.640152, 0.441943])
#q2 = quaternion.Quaternion([0.874, -0.170, -0.033, 0.455])

t1 = numpy.array(inv_rotations.qu2om(q1)).dot([1.0, 0.0, 0.0])
t2 = numpy.linalg.solve(inv_rotations.qu2om(q2), [1.0, 0.0, 0.0])

print numpy.array(inv_rotations.qu2om(q1)).dot([1.0, 0.0, 0.0])
print numpy.array(inv_rotations.qu2om(q2)).dot([1.0, 0.0, 0.0])
print numpy.linalg.solve(inv_rotations.qu2om(q2), [1.0, 0.0, 0.0])