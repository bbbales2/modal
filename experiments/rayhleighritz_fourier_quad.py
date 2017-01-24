#%%
import numpy
import sympy
import time
import scipy
import matplotlib.pyplot as plt

x, y = sympy.symbols('x y')

sympy.sympify("sin(pi * x * n)").evalf(subs = { x : 1.0, 'n' : 0.1111 })

Nv = [sympy.sympify("sin(pi * x * n)"), sympy.sympify("cos(pi * x * n)")]

gx = [n.subs(x, x).subs('n', 'n') for n in Nv]
gy = [n.subs(x, y).subs('n', 'm') for n in Nv]

f = []
for i in range(1, 5):
    for j in range(1, 5):
        ftmp = [gx[0] * gy[0], gx[1] * gy[0], gx[1] * gy[1], gx[0] * gy[1]]

        for ft in ftmp:
            f.append(ft.subs('n', i).subs('m', j))

B = []

zero = sympy.sympify('0')

f = [gx[0] * gy[0], gx[1] * gy[0], gx[1] * gy[1], gx[0] * gy[1]]

fxs = []
fys = []
for k in range(len(f)):
    fxs.append(sympy.diff(f[k], x))
    fys.append(sympy.diff(f[k], y))

basis = []
for i in range(1, 5):
    for j in range(1, 5):
        for k in range(0, 4):
            basis.append((i, j, k))

basis.append((0, 0, 2))

ff = numpy.zeros((len(f), len(f)), dtype = 'object')
#[sympy.lambdify((x, y, 'n', 'm'), fe) for fe in f]

fdf = numpy.zeros((len(f), len(f), 2, 2), dtype = 'object')
fdf2 = numpy.zeros((len(f), len(f), 2, 2), dtype = 'object')
vs = [x, y]
for i in range(len(f)):
    for j in range(len(f)):
        for ii in range(2):
            for jj in range(2):
                df1 = sympy.diff(f[i], vs[ii]).subs('n', 'n0').subs('m', 'm0')
                df2 = sympy.diff(f[j], vs[jj]).subs('n', 'n1').subs('m', 'm1')

                fdf[i, j, ii, jj] = sympy.lambdify((x, y, 'n0', 'm0', 'n1', 'm1'), df1 * df2)
                fdf2[i, j, ii, jj] = df1 * df2

        ff[i, j] = sympy.lambdify((x, y, 'n0', 'm0', 'n1', 'm1'), f[i].subs('n', 'n0').subs('m', 'm0') * f[j].subs('n', 'n1').subs('m', 'm1'))
#%%
t = sympy.sympify('sin(x) + b')
t2 = sympy.lambdify((x, 'b'), t)
#%%
dp = numpy.zeros((len(basis), len(basis), 2, 2))
pv = numpy.zeros((len(basis), len(basis)))

tmp = time.time()
for i in range(len(basis)):
    for j in range(len(basis)):
        n0, m0, k0 = basis[i]
        n1, m1, k1 = basis[j]

        for ii in range(2):
            for jj in range(2):
                t = lambda x, y : fdf[k0, k1, ii, jj](x, y, n0, m0, n1, m1)

                dp[i, j, ii, jj], _ = scipy.integrate.dblquad(t, 0.0, 1.0, lambda x : 0.0, lambda x : 1.0)

        t = lambda x, y : ff[k0, k1](x, y, n0, m0, n1, m1)

        pv[i, j], _ = scipy.integrate.dblquad(t, 0.0, 1.0, lambda x : 0.0, lambda x : 1.0)
    print i

print "Building products {0}!".format(time.time() - tmp)

#%%

import numpy
import time
import scipy

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

tmp = time.time()

K = numpy.zeros((2 * len(basis), 2 * len(basis)))
for i in range(len(basis)):
    for j in range(len(basis)):
        c = numpy.array([[c11 * dp[i, j, 0, 0] + c44 * dp[i, j, 1, 1], c44 * dp[i, j, 1, 0] + c12 * dp[i, j, 0, 1]],
                        [c12 * dp[i, j, 1, 0] + c44 * dp[i, j, 0, 1], c44 * dp[i, j, 0, 0] + c11 * dp[i, j, 1, 1]]])

        K[2 * i, 2 * j] += c[0, 0]
        K[2 * i + 1, 2 * j] += c[1, 0]
        K[2 * i, 2 * j + 1] += c[0, 1]
        K[2 * i + 1, 2 * j + 1] += c[1, 1]
print "Building stiffness {0}".format(time.time() - tmp)

import matplotlib.pyplot as plt

tmp = time.time()

M = numpy.zeros((2 * len(basis), 2 * len(basis)))

for i in range(len(basis)):
    for j in range(len(basis)):
        M[2 * i, 2 * j] = p * pv[i, j]
        M[2 * i + 1, 2 * j + 1] = p * pv[i, j]
print "Building mass {0}".format(time.time() - tmp)

tmp = time.time()
eigs, evecs = scipy.linalg.eigh(K, M)
print "Solving for eigenvalues {0}!".format(time.time() - tmp)
print "\nEigs!"
for eig in sorted(numpy.real(eigs))[0:30]:
    print "{0:.10f}".format(eig)