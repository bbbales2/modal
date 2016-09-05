#%%
import numpy
import matplotlib.pyplot as plt

def fder(C, a, b, y):
    K = numpy.zeros((6, 6))
    dKdQ = numpy.zeros((6, 6, 3, 3))

    Q = numpy.array([[numpy.cos(b) * numpy.cos(y), -numpy.cos(b) * numpy.sin(y), numpy.sin(b)],
                     [numpy.cos(y) * numpy.sin(a) * numpy.sin(b) + numpy.cos(a) * numpy.sin(y), numpy.cos(a) * numpy.cos(y) - numpy.sin(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(b) * numpy.sin(a)],
                     [-numpy.cos(a) * numpy.cos(y) * numpy.sin(b) + numpy.sin(a) * numpy.sin(y), numpy.cos(y) * numpy.sin(a) + numpy.cos(a) * numpy.sin(b) * numpy.sin(y), numpy.cos(a) * numpy.cos(b)]])

    dQda = numpy.array([[0, 0, 0],
            [numpy.cos(a) * numpy.cos(y) * numpy.sin(b) - numpy.sin(a) * numpy.sin(y), -numpy.cos(y) * numpy.sin(a) - numpy.cos(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(a) * numpy.cos(b)],
            [numpy.cos(y) * numpy.sin(a) * numpy.sin(b) + numpy.cos(a) * numpy.sin(y), numpy.cos(a) * numpy.cos(y) - numpy.sin(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(b) * numpy.sin(a)]])

    dQdb = numpy.array([[-numpy.cos(y) * numpy.sin(b), numpy.sin(b) * numpy.sin(y), numpy.cos(b)],
            [numpy.cos(b) * numpy.cos(y) * numpy.sin(a), -numpy.cos(b) * numpy.sin(a) * numpy.sin(y), numpy.sin(a) * numpy.sin(b)],
            [-numpy.cos(a) * numpy.cos(b) * numpy.cos(y), numpy.cos(a) * numpy.cos(b) * numpy.sin(y), -numpy.cos(a) * numpy.sin(b)]])

    dQdy = numpy.array([[-numpy.cos(b) * numpy.sin(y), -numpy.cos(b) * numpy.cos(y), 0],
            [numpy.cos(a) * numpy.cos(y) - numpy.sin(a) * numpy.sin(b) * numpy.sin(y), -numpy.cos(y) * numpy.sin(a) * numpy.sin(b) - numpy.cos(a) * numpy.sin(y), 0],
            [numpy.cos(y) * numpy.sin(a) + numpy.cos(a) * numpy.sin(b) * numpy.sin(y), numpy.cos(a) * numpy.cos(y) * numpy.sin(b) - numpy.sin(a) * numpy.sin(y), 0]])

    for i in range(3):
        for j in range(3):
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
            K[i][j + 3] *= 2.0
            dKdQ[i][j + 3] *= 2.0

    Crot = K.dot(C.dot(K.T))
    dCrotdQ = numpy.zeros((6, 6, 3, 3))
    for i in range(3):
        for j in range(3):
            dCrotdQ[:, :, i, j] = dKdQ[:, :, i, j].dot(C.dot(K.T)) + K.dot(C.dot(dKdQ[:, :, i, j].T))

    dCrotdas = numpy.zeros((6, 6, 3))

    for i in range(6):
        for j in range(6):
            dCrotdas[i, j, 0] = (dCrotdQ[i, j] * dQda).flatten().sum()
            dCrotdas[i, j, 1] = (dCrotdQ[i, j] * dQdb).flatten().sum()
            dCrotdas[i, j, 2] = (dCrotdQ[i, j] * dQdy).flatten().sum()

    return Crot, dCrotdas

C = numpy.array([[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                 [1.0, 2.0, 1.0, 0.0, 0.0, 0.0],
                 [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

a1 = 0.01
a2 = a1 * 1.0001

Crot1, dCrot1 = fder(C, 0.0, a1, 0.0)
Crot2, dCrot2 = fder(C, 0.0, a2, 0.0)

drv = (Crot2 - Crot1) / (a2 - a1)

plt.imshow(drv, interpolation = 'None')
plt.show()

plt.imshow(dCrot1[:, :, 1], interpolation = 'NONE')
plt.show()

print drv.flatten()
print dCrot1[:, :, 1].flatten()