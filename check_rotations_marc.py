#%%
import os
import numpy

os.chdir('/home/bbales2/modal')

from rotations import symmetry
from rotations import quaternion

import itertools
# get the equivalent passive rotation in the fundamental zone (active post multiplies symmetry operator)

#q = [0.98849515554696088, -0.0026722254086214156, 0.015326741589590076, 0.15045024979640861]
#q = (0.981, 0.0, 0.0, -0.1908)
q = [0.594756, -0.202874, 0.640152, 0.441943]
q = q / numpy.linalg.norm(q)

q1 = quaternion.Quaternion(q)

#q = [0.98849515554696088, -0.0026722254086214156, -0.015326741589590076, 0.15045024979640861]
#q = (0.988, 0.0, -0.01, 0.152)
q = [0.928, -0.129, -0.315, -0.141]
q = q / numpy.linalg.norm(q)

q2 = quaternion.Quaternion(q)

cubicSym = symmetry.Symmetry.Cubic.quOperators()
orthoSym = symmetry.Symmetry.Orthorhombic.quOperators()

miso = 180.0#q1 * q2.conjugate()
misoa = None

for i in range(len(cubicSym)):
    qa = cubicSym[i] * q1
    if qa.wxyz[0] < 0.0:
        qa.wxyz[0] *= -1
        qa.wxyz[1] *= -1
        qa.wxyz[2] *= -1
        qa.wxyz[3] *= -1
    
    for k in range(len(cubicSym)):
        qb = cubicSym[k] * q2        
        
        if qb.wxyz[0] < 0.0:
            qb.wxyz[0] *= -1
            qb.wxyz[1] *= -1
            qb.wxyz[2] *= -1
            qb.wxyz[3] *= -1
            
        qasb1 = qa * qb.conjugate()
        qasb2 = qb * qa.conjugate()
        
        if qasb1.wxyz[0] < 0.0:
            qasb1.wxyz[0] *= -1
            qasb1.wxyz[1] *= -1
            qasb1.wxyz[2] *= -1
            qasb1.wxyz[3] *= -1
            
        if qasb2.wxyz[0] < 0.0:
            qasb2.wxyz[0] *= -1
            qasb2.wxyz[1] *= -1
            qasb2.wxyz[2] *= -1
            qasb2.wxyz[3] *= -1
        
        a1 = 2 * numpy.arccos(qasb1.wxyz[0]) * 180 / numpy.pi
        a2 = 2 * numpy.arccos(qasb2.wxyz[0]) * 180 / numpy.pi
        
        if a1 < miso:
            miso = a1
            misoa = qasb1
            
        if a2 < miso:
            miso = a2
            misoa = qasb2
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