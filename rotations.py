#%%
import numpy

C = numpy.zeros((3, 3))
Q = numpy.zeros((3, 3))
K = numpy.zeros((6, 6))
dKdQ = numpy.zeros((6, 6, 3, 3))

dQda = {{0, 0, 0}, {Cos[a] Cos[y] Sin[b] - Sin[a] Sin[y], -Cos[y] Sin[a] -
   Cos[a] Sin[b] Sin[y], -Cos[a] Cos[b]}, {Cos[y] Sin[a] Sin[b] +
   Cos[a] Sin[y],
  Cos[a] Cos[y] - Sin[a] Sin[b] Sin[y], -Cos[b] Sin[a]}}

dQdb = {{-Cos[y] Sin[b], Sin[b] Sin[y],
  Cos[b]}, {Cos[b] Cos[y] Sin[a], -Cos[b] Sin[a] Sin[y],
  Sin[a] Sin[b]}, {-Cos[a] Cos[b] Cos[y],
  Cos[a] Cos[b] Sin[y], -Cos[a] Sin[b]}}

dQdy = {{-Cos[b] Sin[y], -Cos[b] Cos[y],
  0}, {Cos[a] Cos[y] - Sin[a] Sin[b] Sin[y], -Cos[y] Sin[a] Sin[b] -
   Cos[a] Sin[y], 0}, {Cos[y] Sin[a] + Cos[a] Sin[b] Sin[y],
  Cos[a] Cos[y] Sin[b] - Sin[a] Sin[y], 0}}

for i in range(3):
    for j in range(3):
        #print "[{0}, {1}]".format(i, j)
        #print "[{0}, {1}] [{2}, {3}]".format(i, (j + 1) % 3, i, (j + 2) % 3)
        #print "[{0}, {1}] [{2}, {3}]".format((i + 1) % 3, j, (i + 2) % 3, j)
        #print "[{0}, {1}] [{2}, {3}], [{4}, {5}] [{6}, {7}]".format((i + 1) % 3, (j + 1) % 3, (i + 2) % 3, (j + 2) % 3, (i + 1) % 3, (j + 2) % 3, (i + 2) % 3, (j + 1) % 3)
        dKdQ[i, j, i, j] = 2.0 * Q[i, j]
        dKdQ[i, j + 3, i, (j + 1) % 3] = Q[i, (j + 2) % 3]
        dKdQ[i, j + 3, i, (j + 2) % 3] = Q[i, (j + 1) % 3]
        dKdQ[i + 3, j, (i + 1) % 3, j] = Q[(i + 2) % 3, j]
        dKdQ[i + 3, j, (i + 2) % 3, j] = Q[(i + 1) % 3, j]
        dKdQ[i + 3, j + 3, (i + 1) % 3, (j + 1) % 3] = Q[(i + 2) % 3, (j + 2) % 3]
        dKdQ[i + 3, j + 3, (i + 2) % 3, (j + 2) % 3] = Q[(i + 1) % 3, (j + 1) % 3]
        dKdQ[i + 3, j + 3, (i + 1) % 3, (j + 2) % 3] = Q[(i + 2) % 3, (j + 1) % 3]
        dKdQ[i + 3, j + 3, (i + 2) % 3, (j + 1) % 3] = Q[(i + 1) % 3, (j + 2) % 3]
        K[i+0][j+0] = Q[i][j] * Q[i][j]
        K[i+0][j+3] = Q[i][(j+1)%3] * Q[i][(j+2)%3]
        K[i+3][j+0] = Q[(i+1)%3][j] * Q[(i+2)%3][j]
        K[i+3][j+3] = Q[(i+1)%3][(j+1)%3] * Q[(i+2)%3][(j+2)%3] + Q[(i+1)%3][(j+2)%3] * Q[(i+2)%3][(j+1)%3]


for i in range(3):
    for j in range(3):
        K[i+0][j+3] *= 2.0
        dKdQ[i+0][j+3] *= 2.0

Crot = K.dot(C.dot(K.T))
dCrotdQ = numpy.zeros((6, 6, 3, 3))
for i in range(3):
    for j in range(3):
        dCrotdQ[:, :, i, j] = dKdQ[:, :, i, j].dot(C.dot(K.T)) + K.dot(C.dot(dKdQ[:, :, i, j].T))
