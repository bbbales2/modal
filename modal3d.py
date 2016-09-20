#%%
import numpy
import scipy
import matplotlib.pyplot as plt
import pickle
import time
import pysparse
import scipy
import sympy

N = 10

p = 2700
young = 6.8
poisson = 0.36

mu0 = young / (2.0 * (1.0 + poisson))
lambda0 = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

c11 = lambda0 + 2 * mu0
c12 = lambda0
c44 = mu0

X = 1.0
Y = 1.0
Z = 1.0

q = X / N
w = Y / N
r = Z / N

Dd = numpy.array([[c11, c12, c12, 0, 0, 0],
                  [c12, c11, c12, 0, 0, 0],
                  [c12, c12, c11, 0, 0, 0],
                  [0, 0, 0, c44, 0, 0],
                  [0, 0, 0, 0, c44, 0],
                  [0, 0, 0, 0, 0, c44]])

x, y, z = sympy.symbols('x y z')

Nv = [(1 - x) / 2, (1 + x) / 2]

gx = [n.subs(x, x) for n in Nv]
gy = [n.subs(x, y) for n in Nv]
gz = [n.subs(x, z) for n in Nv]

f = [gx[0] * gy[0] * gz[0], gx[1] * gy[0] * gz[0], gx[1] * gy[1] * gz[0], gx[0] * gy[1] * gz[0],
     gx[0] * gy[0] * gz[1], gx[1] * gy[0] * gz[1], gx[1] * gy[1] * gz[1], gx[0] * gy[1] * gz[1]]

B = []

pts = [-numpy.sqrt(3.0 / 5.0), 0, numpy.sqrt(3.0 / 5.0)]
weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

dpts = [-numpy.sqrt(1.0 / 3.0), numpy.sqrt(1.0 / 3.0)]
dweights = [1.0, 1.0]

dpts = [-numpy.sqrt(3.0 / 5.0), 0, numpy.sqrt(3.0 / 5.0)]
dweights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

dfx = numpy.zeros((len(f), len(dpts), len(dpts), len(dpts)))
dfy = numpy.zeros((len(f), len(dpts), len(dpts), len(dpts)))
dfz = numpy.zeros((len(f), len(dpts), len(dpts), len(dpts)))

fv = numpy.zeros((len(f), len(pts), len(pts), len(pts)))

for l in range(len(f)):
    fx = sympy.diff(f[l], x)
    fy = sympy.diff(f[l], y)
    fz = sympy.diff(f[l], z)

    for i in range(len(dpts)):
        for j in range(len(dpts)):
            for k in range(len(dpts)):
                dfx[l, i, j, k] = fx.evalf(subs = { x : dpts[i], y : dpts[j], z : dpts[k] })
                dfy[l, i, j, k] = fy.evalf(subs = { x : dpts[i], y : dpts[j], z : dpts[k] })
                dfz[l, i, j, k] = fz.evalf(subs = { x : dpts[i], y : dpts[j], z : dpts[k] })

    for i in range(len(pts)):
        for j in range(len(pts)):
            for k in range(len(pts)):
                fv[l, i, j, k] = f[l].evalf(subs = { x : pts[i], y : pts[j], z : pts[k] })

ke = numpy.zeros((len(f), len(f), 3, 3))

me = numpy.zeros((len(f), len(f)))
for a in range(len(f)):
    for b in range(len(f)):
        for i in range(len(dpts)):
            for j in range(len(dpts)):
                for k in range(len(dpts)):
                    Ba = numpy.array([[dfx[a, i, j, k], 0.0, 0.0],
                                      [0.0, dfy[a, i, j, k], 0.0],
                                      [0.0, 0.0, dfz[a, i, j, k]],
                                      [0.0, dfz[a, i, j, k], dfy[a, i, j, k]],
                                      [dfz[a, i, j, k], 0.0, dfx[a, i, j, k]],
                                      [dfy[a, i, j, k], dfx[a, i, j, k], 0.0]])

                    Bb = numpy.array([[dfx[b, i, j, k], 0.0, 0.0],
                                      [0.0, dfy[b, i, j, k], 0.0],
                                      [0.0, 0.0, dfz[b, i, j, k]],
                                      [0.0, dfz[b, i, j, k], dfy[b, i, j, k]],
                                      [dfz[b, i, j, k], 0.0, dfx[b, i, j, k]],
                                      [dfy[b, i, j, k], dfx[b, i, j, k], 0.0]])

                    ke[a, b] += dweights[i] * dweights[j] * dweights[k] * Ba.T.dot(Dd.dot(Bb))

        for i in range(len(pts)):
            for j in range(len(pts)):
                for k in range(len(pts)):
                    me[a, b] += weights[i] * weights[j] * weights[k] * fv[a, i, j, k] * fv[b, i, j, k]

ke *= 20#w * q * r / 8.0
me *= p * w * q * r / 8.0

#%%
xs = numpy.linspace(0.0, X, N + 1)
ys = numpy.linspace(0.0, Y, N + 1)
zs = numpy.linspace(0.0, Z, N + 1)

tmp = time.time()
n2d_gn1d = {}
gn1d_n1d = {}
dof = 0
for i in range(N + 1):
    for j in range(N + 1):
        for k in range(N + 1):
            n2d_gn1d[(i, j, k)] = i * (N + 1) * (N + 1) + j * (N + 1) + k

            #if i == 0 or j == 0 or i >= N or j >= N:
            #    gn1d_n1d[(i * (N + 1) + j, 0)] = -1
            #    gn1d_n1d[(i * (N + 1) + j, 1)] = -1
            #else:
            if True:
                gn1d_n1d[(i * (N + 1) * (N + 1) + j * (N + 1) + k, 0)] = dof
                dof += 1
                gn1d_n1d[(i * (N + 1) * (N + 1) + j * (N + 1) + k, 1)] = dof
                dof += 1
                gn1d_n1d[(i * (N + 1) * (N + 1) + j * (N + 1) + k, 2)] = dof
                dof += 1

rowsK = []
colsK = []
dataK = []

rowsM = []
colsM = []
dataM = []

e2d_e1d = {}
for i in range(N):
    for j in range(N):
        for k in range(N):
            e2d_e1d[(i, j, k)] = i * N * N + j * N + k

for ei, ej, ek in e2d_e1d:
    nodes = []
    for ni, nj, nk in zip([ei + 1, ei + 1, ei, ei, ei + 1, ei + 1, ei, ei], [ej, ej + 1, ej + 1, ej, ej, ej + 1, ej + 1, ej], [ek, ek, ek, ek, ek + 1, ek + 1, ek + 1, ek + 1]):
        for i in range(3):
            nodes.append(gn1d_n1d[(n2d_gn1d[(ni, nj, nk)], i)])

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if node1 < node2 and node1 != -1 and node2 != -1:
                if node1 != node2:
                    rowsK.append(node1)
                    colsK.append(node2)

                    rowsK.append(node2)
                    colsK.append(node1)

                    dataK.append(ke[i / 3, j / 3, i % 3, j % 3])
                    dataK.append(ke[i / 3, j / 3, i % 3, j % 3])

                if (i % 2) == (j % 2):
                    rowsM.append(node1)
                    colsM.append(node2)

                    rowsM.append(node2)
                    colsM.append(node1)

                    dataM.append(me[i / 3, j / 3])
                    dataM.append(me[i / 3, j / 3])

            if node1 == node2 and node1 != -1 and node2 != -1:
                rowsK.append(node1)
                colsK.append(node2)
                dataK.append(ke[i / 3, j / 3, i % 3, j % 3])

                rowsM.append(node1)
                colsM.append(node2)
                dataM.append(me[i / 3, j / 3])


            #if node1 >= node2 and node1 != -1 and node2 != -1:
                #K[node1, node2] += ke[i / 2, j / 2, i % 2, j % 2]
            #    K1[node1, node2] += ke[i / 2, j / 2, i % 2, j % 2]

            #    if (i % 2) == (j % 2):
                    #M[node1, node2] += me[i / 2, j / 2]
             #       M1[node1, node2] += me[i / 2, j / 2]



K = scipy.sparse.coo_matrix((dataK, (rowsK, colsK)), shape = (dof, dof))
M = scipy.sparse.coo_matrix((dataM, (rowsM, colsM)), shape = (dof, dof))

#Kdiag = numpy.diag(numpy.diag(K))
#K = (K - Kdiag).T + K

#Mdiag = numpy.diag(numpy.diag(M))
#M = (M - Mdiag).T + M

print "Assembly: ", time.time() - tmp

#%%

tmp = time.time()
eigs, evecs = scipy.sparse.linalg.eigsh(K, 30, M = M, sigma = 0.0)
print time.time() - tmp

print eigs

#%%
[ 0.00770872  0.00770872  0.01415904  0.01415904  0.01415904  0.014525
  0.014525    0.014525    0.01865016  0.01865016  0.01900606  0.02154008
  0.02154008  0.02154008  0.02242065  0.02242065  0.02242065  0.02916514
  0.02986575  0.02986575  0.02986575  0.03071747  0.03071747  0.03142756
  0.03142756  0.03142756  0.03840492  0.03840492  0.03840492  0.04266346
  0.04266346  0.04266346  0.05215902  0.05215902  0.05215902  0.05402425
  0.05402425  0.05402425  0.05577358  0.05577358  0.05577358  0.0563878
  0.0563878   0.05647535  0.05671449  0.06662193  0.06662193  0.070873
  0.070873    0.070873  ]

#%%
K = K.todense()
M = M.todense()
#%%
f = open('tmp', 'r')
K2, M2 = pickle.load(f)
f.close()

K2 = K2.todense()
M2 = M2.todense()

a = K2[0:100, 0:100] - K[0:100, 0:100]

plt.imshow(a, interpolation = 'NONE')
plt.colorbar()
plt.show()

plt.imshow(K[0:50, 0:50], interpolation = 'NONE')
plt.show()

plt.imshow(K2[0:50, 0:50], interpolation = 'NONE')
plt.show()
