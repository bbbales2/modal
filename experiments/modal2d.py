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

q = X / N
w = Y / N

Dd = numpy.array([[c11, c12, 0],
                  [c12, c11, 0],
                  [0, 0, c44]])

x, y = sympy.symbols('x y')

Nv = [(1 - x) / 2, (1 + x) / 2]

gx = [n.subs(x, x) for n in Nv]
gy = [n.subs(x, y) for n in Nv]

f = [gx[0] * gy[0], gx[1] * gy[0], gx[1] * gy[1], gx[0] * gy[1]]

B = []

zero = sympy.sympify('0')

pts = [-numpy.sqrt(3.0 / 5.0), 0, numpy.sqrt(3.0 / 5.0)]
weights = [5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]

pts = [-numpy.sqrt(1.0 / 3.0), numpy.sqrt(1.0 / 3.0)]
weights = [1.0, 1.0]

dpts = [-numpy.sqrt(1.0 / 3.0), numpy.sqrt(1.0 / 3.0)]
dweights = [1.0, 1.0]

dfx = numpy.zeros((len(f), len(dpts), len(dpts)))
dfy = numpy.zeros((len(f), len(dpts), len(dpts)))

fv = numpy.zeros((len(f), len(pts), len(pts)))

for k in range(len(f)):
    fx = sympy.diff(f[k], x)
    fy = sympy.diff(f[k], y)

    for i in range(len(dpts)):
        for j in range(len(dpts)):
            dfx[k, i, j] = fx.evalf(subs = { x : dpts[i], y : dpts[j] })
            dfy[k, i, j] = fy.evalf(subs = { x : dpts[i], y : dpts[j] })

    for i in range(len(pts)):
        for j in range(len(pts)):
            fv[k, i, j] = f[k].evalf(subs = { x : pts[i], y : pts[j] })

ke = numpy.zeros((len(f), len(f), 2, 2))

me = numpy.zeros((len(f), len(f)))
for a in range(len(f)):
    for b in range(len(f)):

        fxa = sympy.diff(f[a], x)
        fya = sympy.diff(f[a], y)

        fxb = sympy.diff(f[b], x)
        fyb = sympy.diff(f[b], y)

        for i in range(len(dpts)):
            for j in range(len(dpts)):
                Ba = numpy.array([[dfx[a, i, j], 0.0],
                                  [0.0, dfy[a, i, j]],
                                  [dfy[a, i, j], dfx[a, i, j]]])

                Bb = numpy.array([[dfx[b, i, j], 0.0],
                                  [0.0, dfy[b, i, j]],
                                  [dfy[b, i, j], dfx[b, i, j]]])

                ke[a, b] += dweights[i] * dweights[j] * Ba.T.dot(Dd.dot(Bb))

        for i in range(len(pts)):
            for j in range(len(pts)):
                me[a, b] += weights[i] * weights[j] * fv[a, i, j] * fv[b, i, j]

me *= p * w * q / 4.0
#%%
#ke1 *= w * q / (4.0)

ke1 = numpy.array([
  [
    [
      [(c44 * q)/(3 * w) + (c11 * w)/(3 * q), (c12 + c44)/4],
      [(c12 + c44)/4, (c11 * q)/(3 * w) + (c44 * w)/(3 * q)]
    ],
    [
      [-((c44 * q)/(3 * w)) + (c11 * w)/(6 * q), 0.25 * (-c12 + c44)],
      [(c12 - c44)/4, -((c11 * q)/(3 * w)) + (c44 * w)/(6 * q)]
    ],
    [
      [-((c44 * q)/(6 * w)) - (c11 * w)/(6 * q), 0.25 * (-c12 - c44)],
      [0.25 * (-c12 - c44), -((c11 * q)/(6 * w)) - (c44 * w)/(6 * q)]
    ],
    [
      [(c44 * q)/(6 * w) - (c11 * w)/(3 * q), (c12 - c44)/4],
      [0.25 * (-c12 + c44), (c11 * q)/(6 * w) - (c44 * w)/(3 * q)]
    ]
  ],
  [
    [
      [-((c44 * q)/(3 * w)) + (c11 * w)/(6 * q), (c12 - c44)/4],
      [0.25 * (-c12 + c44), -((c11 * q)/(3 * w)) + (c44 * w)/(6 * q)]
    ],
    [
      [(c44 * q)/(3 * w) + (c11 * w)/(3 * q), 0.25 * (-c12 - c44)],
      [0.25 * (-c12 - c44), (c11 * q)/(3 * w) + (c44 * w)/(3 * q)]
    ],
    [
      [(c44 * q)/(6 * w) - (c11 * w)/(3 * q), 0.25 * (-c12 + c44)],
      [(c12 - c44)/4, (c11 * q)/(6 * w) - (c44 * w)/(3 * q)]
    ],
    [
      [-((c44 * q)/(6 * w)) - (c11 * w)/(6 * q), (c12 + c44)/4],
      [(c12 + c44)/4, -((c11 * q)/(6 * w)) - (c44 * w)/(6 * q)]
    ]
  ],
  [
    [
      [-((c44 * q)/(6 * w)) - (c11 * w)/(6 * q), 0.25 * (-c12 - c44)],
      [0.25 * (-c12 - c44), -((c11 * q)/(6 * w)) - (c44 * w)/(6 * q)]
    ],
    [
      [(c44 * q)/(6 * w) - (c11 * w)/(3 * q), (c12 - c44)/4],
      [0.25 * (-c12 + c44), (c11 * q)/(6 * w) - (c44 * w)/(3 * q)]
    ],
    [
      [(c44 * q)/(3 * w) + (c11 * w)/(3 * q), (c12 + c44)/4],
      [(c12 + c44)/4, (c11 * q)/(3 * w) + (c44 * w)/(3 * q)]
    ],
    [
      [-((c44 * q)/(3 * w)) + (c11 * w)/(6 * q), 0.25 * (-c12 + c44)],
      [(c12 - c44)/4, -((c11 * q)/(3 * w)) + (c44 * w)/(6 * q)]
    ]
  ],
  [
    [
      [(c44 * q)/(6 * w) - (c11 * w)/(3 * q), 0.25 * (-c12 + c44)],
      [(c12 - c44)/4, (c11 * q)/(6 * w) - (c44 * w)/(3 * q)]
    ],
    [
      [-((c44 * q)/(6 * w)) - (c11 * w)/(6 * q), (c12 + c44)/4],
      [(c12 + c44)/4, -((c11 * q)/(6 * w)) - (c44 * w)/(6 * q)]
    ],
    [
      [-((c44 * q)/(3 * w)) + (c11 * w)/(6 * q), (c12 - c44)/4],
      [0.25 * (-c12 + c44), -((c11 * q)/(3 * w)) + (c44 * w)/(6 * q)]
    ],
    [
      [(c44 * q)/(3 * w) + (c11 * w)/(3 * q), 0.25 * (-c12 - c44)],
      [0.25 * (-c12 - c44), (c11 * q)/(3 * w) + (c44 * w)/(3 * q)]
    ]
  ]
])

print "ke1: ", ke1[0, 1]
print "ke: ", ke[0, 1]
#%%
me1 = p * numpy.array([
  [(q * w)/9, (q * w)/18, (q * w)/36, (q * w)/18],
  [(q * w)/18, (q * w)/9, (q * w)/18, (q * w)/36],
  [(q * w)/36, (q * w)/18, (q * w)/9, (q * w)/18],
  [(q * w)/18, (q * w)/36, (q * w)/18, (q * w)/9]
])

xs = numpy.linspace(0.0, X, N + 1)
ys = numpy.linspace(0.0, Y, N + 1)

tmp = time.time()
n2d_gn1d = {}
gn1d_n1d = {}
dof = 0
for i in range(N + 1):
    for j in range(N + 1):
        n2d_gn1d[(i, j)] = i * (N + 1) + j

        #if i == 0 or j == 0 or i >= N or j >= N:
        #    gn1d_n1d[(i * (N + 1) + j, 0)] = -1
        #    gn1d_n1d[(i * (N + 1) + j, 1)] = -1
        #else:
        if True:
            gn1d_n1d[(i * (N + 1) + j, 0)] = dof
            dof += 1
            gn1d_n1d[(i * (N + 1) + j, 1)] = dof
            dof += 1

K = numpy.zeros((dof, dof))
K1 = pysparse.spmatrix.ll_mat_sym(dof)
M = numpy.zeros((dof, dof))
M1 = pysparse.spmatrix.ll_mat_sym(dof)

rowsK = []
colsK = []
dataK = []

rowsM = []
colsM = []
dataM = []

e2d_e1d = {}
for i in range(N):
    for j in range(N):
        e2d_e1d[(i, j)] = i * N + j

for ei, ej in e2d_e1d:
    nodes = []
    for ni, nj in zip([ei + 1, ei + 1, ei, ei], [ej, ej + 1, ej + 1, ej]):
        for i in range(2):
            nodes.append(gn1d_n1d[(n2d_gn1d[(ni, nj)], i)])

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            rowsK.append(node1)
            colsK.append(node2)
            dataK.append(ke[i / 2, j / 2, i % 2, j % 2])

            if (i % 2) == (j % 2):
                rowsM.append(node1)
                colsM.append(node2)
                dataM.append(me[i / 2, j / 2])

            if node1 >= node2 and node1 != -1 and node2 != -1:
                #K[node1, node2] += ke[i / 2, j / 2, i % 2, j % 2]
                K1[node1, node2] += ke[i / 2, j / 2, i % 2, j % 2]

                if (i % 2) == (j % 2):
                    #M[node1, node2] += me[i / 2, j / 2]
                    M1[node1, node2] += me[i / 2, j / 2]

K = scipy.sparse.coo_matrix((dataK, (rowsK, colsK)), shape = (dof, dof))
M = scipy.sparse.coo_matrix((dataM, (rowsM, colsM)), shape = (dof, dof))

#Kdiag = numpy.diag(numpy.diag(K))
#K = (K - Kdiag).T + K

#Mdiag = numpy.diag(numpy.diag(M))
#M = (M - Mdiag).T + M

print "Assembly: ", time.time() - tmp

#eigs, evecs = scipy.linalg.eigh(K, M)

#print eigs[:50]
#%%
K1 = K1.to_sss()
M1 = M1.to_sss()
#%%
tmp = time.time()
S1 = pysparse.precon.ssor(K1)
kconv, lmbd, Q, it, it_inner = pysparse.jdsym.jdsym(K1, M1, S1, 30, 0.0, 1e-4, 1000, pysparse.itsolvers.pcg)
print time.time() - tmp
#%%
K = K.tocsc()
M = M.tocsc()
#%%
tmp = time.time()
eigs, evecs = scipy.sparse.linalg.eigsh(K, 30, M = M, sigma = 0.0)
print time.time() - tmp
#%%
eigsCheck = numpy.array([ 0.01450785,  0.01838007,  0.01866904,  0.01866904,  0.02820598,  0.03850401,
  0.03850401,  0.05068253,  0.05488751,  0.07303677,  0.07303677,  0.07480172,
  0.08005152,  0.08232707,  0.11032394,  0.11032394,  0.1104149,   0.13285177,
  0.13285177,  0.13680392,  0.15017603,  0.16495508,  0.1730885,   0.17486998,
  0.17486998,  0.18709228,  0.19278279,  0.19278279,  0.23333533,  0.23333533,
  0.23494067,  0.24348693,  0.24348693,  0.25484549,  0.26770504,  0.27156861,
  0.2927992,   0.29830237,  0.31076889,  0.31076889,  0.3199193,   0.35205066,
  0.35243956,  0.36745689, 0.36745689,  0.37602505,  0.37602505,  0.40082352,
  0.40747666,  0.40747666])

eigsCheck2 = numpy.array([ 0.0000000000,
0.0000000000,
0.0000000000,
0.0144283280,
0.0182770452,
0.0185246733,
0.0185246733,
0.0279338108,
0.0379163118,
0.0379163118,
0.0501947146,
0.0534694194,
0.0711413702,
0.0711413702,
0.0731154980,
0.0781361199,
0.0812594193,
0.1053760922,
0.1065283425,
0.1065283425,
0.1276417293,
0.1276417293,
0.1340619977,
0.1470586970,
0.1642580140,
0.1697078687,
0.1699234240,
0.1699234240,
0.1892765492,
0.1892765492])
#%%
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
#%%
#eigs2 = eigs.copy()

#res = numpy.linalg.solve(K, numpy.ones(dof))
#[:, :, 1]
plt.imshow(evecs[0].reshape((N + 1, N + 1, 2))[:, :, 0], interpolation = 'NONE')
plt.show()
#%%
#%%

us = evecs[:, 5].reshape((N, N, 2))

plt.imshow(us[:, :, 0], interpolation = 'NONE')
plt.colorbar()
plt.show()

plt.imshow(us[:, :, 1], interpolation = 'NONE')
plt.colorbar()
plt.show()