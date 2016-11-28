#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy
import time

cpdef inline double polyint(int n, int m, int l):
    cdef double xtmp, ytmp

    if n < 0 or m < 0 or l < 0 or n % 2 > 0 or m % 2 > 0 or l % 2 > 0:
        return 0.0

    xtmp = 2 * 0.5**(n + 1)
    ytmp = 2 * 0.5**(m + 1) * xtmp
    return 2 * 0.5**(l + 1) * ytmp / ((n + 1) * (m + 1) * (l + 1))

cpdef build(int N, double X, double Y, double Z):
    cdef numpy.ndarray[numpy.double_t, ndim = 4] dp, ddpdX, ddpdY, ddpdZ
    cdef numpy.ndarray[numpy.double_t, ndim = 2] pv, dpvdX, dpvdY, dpvdZ
    cdef numpy.ndarray[numpy.int_t, ndim = 2] basis
    cdef numpy.ndarray[numpy.double_t, ndim = 1] Xs, Ys, Zs

    cdef int i, j, k, basisLength, n0, m0, n1, m1, l0, l1

    tmp = time.time()

    basistmp = []

    for i in range(0, N + 1):
        for j in range(0, N + 1):
            for k in range(0, N + 1):
                if i + j + k <= N:
                    basistmp.append((i, j, k))

    basis = numpy.array(basistmp, dtype = 'int')

    basisLength = len(basis)

    dp = numpy.zeros((len(basis), len(basis), 3, 3))
    ddpdX = numpy.zeros((len(basis), len(basis), 3, 3))
    ddpdY = numpy.zeros((len(basis), len(basis), 3, 3))
    ddpdZ = numpy.zeros((len(basis), len(basis), 3, 3))

    pv = numpy.zeros((len(basis), len(basis)))
    dpvdX = numpy.zeros((len(basis), len(basis)))
    dpvdY = numpy.zeros((len(basis), len(basis)))
    dpvdZ = numpy.zeros((len(basis), len(basis)))

    tmp = time.time()

    Xs = numpy.zeros((2 * N + 3))
    Ys = numpy.zeros((2 * N + 3))
    Zs = numpy.zeros((2 * N + 3))

    tmp = time.time()
    for i in range(-1, 2 * N + 2):
        Xs[i + 1] = X**(i)
        Ys[i + 1] = Y**(i)
        Zs[i + 1] = Z**(i)

    for i in range(basisLength):
        for j in range(basisLength):
            n0 = basis[i, 0]
            m0 = basis[i, 1]
            l0 = basis[i, 2]
            n1 = basis[j, 0]
            m1 = basis[j, 1]
            l1 = basis[j, 2]

            dp[i, j, 0, 0] = Xs[n1 + n0] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 2, m1 + m0, l1 + l0) * n0 * n1#
            dp[i, j, 0, 1] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * n0 * m1
            dp[i, j, 0, 2] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * n0 * l1

            dp[i, j, 1, 0] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 2] * polyint(n1 + n0 - 1, m1 + m0 - 1, l1 + l0) * m0 * n1
            dp[i, j, 1, 1] = Xs[n1 + n0 + 2] * Ys[m1 + m0] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0 - 2, l1 + l0) * m0 * m1
            dp[i, j, 1, 2] = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * m0 * l1

            dp[i, j, 2, 0] = Xs[n1 + n0 + 1] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 1] * polyint(n1 + n0 - 1, m1 + m0, l1 + l0 - 1) * l0 * n1
            dp[i, j, 2, 1] = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 1] * Zs[l1 + l0 + 1] * polyint(n1 + n0, m1 + m0 - 1, l1 + l0 - 1) * l0 * m1
            dp[i, j, 2, 2] = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0 ] * polyint(n1 + n0, m1 + m0, l1 + l0 - 2) * l0 * l1

            ddpdX[i, j, 0, 0] =  dp[i, j, 0, 0] * (n1 + n0 - 1) / X
            ddpdX[i, j, 0, 1] =  dp[i, j, 0, 1] * (n1 + n0) / X
            ddpdX[i, j, 0, 2] =  dp[i, j, 0, 2] * (n1 + n0) / X

            ddpdX[i, j, 1, 0] =  dp[i, j, 1, 0] * (n1 + n0) / X
            ddpdX[i, j, 1, 1] =  dp[i, j, 1, 1] * (n1 + n0 + 1) / X
            ddpdX[i, j, 1, 2] =  dp[i, j, 1, 2] * (n1 + n0 + 1) / X

            ddpdX[i, j, 2, 0] =  dp[i, j, 2, 0] * (n1 + n0) / X
            ddpdX[i, j, 2, 1] =  dp[i, j, 2, 1] * (n1 + n0 + 1) / X
            ddpdX[i, j, 2, 2] =  dp[i, j, 2, 2] * (n1 + n0 + 1) / X

            ddpdY[i, j, 0, 0] =  dp[i, j, 0, 0] * (m1 + m0 + 1) / Y
            ddpdY[i, j, 0, 1] =  dp[i, j, 0, 1] * (m1 + m0) / Y
            ddpdY[i, j, 0, 2] =  dp[i, j, 0, 2] * (m1 + m0 + 1) / Y

            ddpdY[i, j, 1, 0] =  dp[i, j, 1, 0] * (m1 + m0) / Y
            ddpdY[i, j, 1, 1] =  dp[i, j, 1, 1] * (m1 + m0 - 1) / Y
            ddpdY[i, j, 1, 2] =  dp[i, j, 1, 2] * (m1 + m0) / Y

            ddpdY[i, j, 2, 0] =  dp[i, j, 2, 0] * (m1 + m0 + 1) / Y
            ddpdY[i, j, 2, 1] =  dp[i, j, 2, 1] * (m1 + m0) / Y
            ddpdY[i, j, 2, 2] =  dp[i, j, 2, 2] * (m1 + m0 + 1) / Y

            ddpdZ[i, j, 0, 0] =  dp[i, j, 0, 0] * (l1 + l0 + 1) / Z
            ddpdZ[i, j, 0, 1] =  dp[i, j, 0, 1] * (l1 + l0 + 1) / Z
            ddpdZ[i, j, 0, 2] =  dp[i, j, 0, 2] * (l1 + l0) / Z

            ddpdZ[i, j, 1, 0] =  dp[i, j, 1, 0] * (l1 + l0 + 1) / Z
            ddpdZ[i, j, 1, 1] =  dp[i, j, 1, 1] * (l1 + l0 + 1) / Z
            ddpdZ[i, j, 1, 2] =  dp[i, j, 1, 2] * (l1 + l0) / Z

            ddpdZ[i, j, 2, 0] =  dp[i, j, 2, 0] * (l1 + l0) / Z
            ddpdZ[i, j, 2, 1] =  dp[i, j, 2, 1] * (l1 + l0) / Z
            ddpdZ[i, j, 2, 2] =  dp[i, j, 2, 2] * (l1 + l0 - 1) / Z

            pv[i, j] = Xs[n1 + n0 + 2] * Ys[m1 + m0 + 2] * Zs[l1 + l0 + 2] * polyint(n1 + n0, m1 + m0, l1 + l0)

            dpvdX[i, j] = pv[i, j] * (n1 + n0 + 1) / X

            dpvdY[i, j] = pv[i, j] * (m1 + m0 + 1) / Y

            dpvdZ[i, j] = pv[i, j] * (l1 + l0 + 1) / Z

    return dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ

cpdef buildRot(C, w, x, y, z):
    # Code stolen from Will Lenthe
    K = numpy.zeros((6, 6))
    dKdQ = numpy.zeros((6, 6, 3, 3))

    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    dQdw = numpy.array([[2 * w, -2.0 * z, 2.0 * y],
                        [2.0 * z, 2 * w, -2.0 * x],
                        [-2.0 * y, 2.0 * x, 2 * w]])

    dQdx = numpy.array([[2 * x, 2.0 * y, 2.0 * z],
                        [2.0 * y, -2.0 * x, -2.0 * w],
                        [2.0 * z, 2.0 * w, -2.0 * x]])

    dQdy = numpy.array([[-2 * y, 2 * x, 2 * w],
                        [2 * x, 2 * y, 2 * z],
                        [-2 * w, 2 * z, -2 * y]])

    dQdz = numpy.array([[-2 * z, -2 * w, 2 * x],
                        [2 * w, -2 * z, 2 * y],
                        [2 * x, 2 * y, 2 * z]])

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

            K[i][j] = Q[i][j] * Q[i][j]
            K[i][j + 3] = Q[i][(j + 1) % 3] * Q[i][(j + 2) % 3]
            K[i + 3][j] = Q[(i + 1) % 3][j] * Q[(i + 2) % 3][j]
            K[i + 3][j + 3] = Q[(i + 1) % 3][(j + 1) % 3] * Q[(i + 2) % 3][(j + 2) % 3] + Q[(i + 1) % 3][(j + 2) % 3] * Q[(i + 2) % 3][(j + 1) % 3]

    for i in range(3):
        for j in range(3):
            K[i][j + 3] *= 2.0
            dKdQ[i][j + 3] *= 2.0

    Crot = K.dot(C.dot(K.T))
    dCrotdQ = numpy.zeros((6, 6, 3, 3))
    for i in range(3):
        for j in range(3):
            dCrotdQ[:, :, i, j] = dKdQ[:, :, i, j].dot(C.dot(K.T)) + K.dot(C.dot(dKdQ[:, :, i, j].T))

    dCrotdw = numpy.zeros((6, 6))
    dCrotdx = numpy.zeros((6, 6))
    dCrotdy = numpy.zeros((6, 6))
    dCrotdz = numpy.zeros((6, 6))

    for i in range(6):
        for j in range(6):
            dCrotdw[i, j] = (dCrotdQ[i, j] * dQdw).flatten().sum()
            dCrotdx[i, j] = (dCrotdQ[i, j] * dQdx).flatten().sum()
            dCrotdy[i, j] = (dCrotdQ[i, j] * dQdy).flatten().sum()
            dCrotdz[i, j] = (dCrotdQ[i, j] * dQdz).flatten().sum()

    return Crot, dCrotdw, dCrotdx, dCrotdy, dCrotdz, K

cpdef buildRot2(C, w, x, y, z):
    # Code stolen from Will Lenthe
    K = numpy.zeros((6, 6))
    dKdQ = numpy.zeros((6, 6, 3, 3))

    Q = numpy.array([[w**2 - (y**2 + z**2) + x**2, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
                     [2.0 * (y * x + w * z), w**2 - (x**2 + z**2) + y**2, 2.0 * (y * z - w * x)],
                     [2.0 * (z * x - w * y), 2.0 * (z * y + w * x), w**2 - (x**2 + y**2) + z**2]])

    dQdw = numpy.array([[2 * w, -2.0 * z, 2.0 * y],
                        [2.0 * z, 2 * w, -2.0 * x],
                        [-2.0 * y, 2.0 * x, 2 * w]])

    dQdx = numpy.array([[2 * x, 2.0 * y, 2.0 * z],
                        [2.0 * y, -2.0 * x, -2.0 * w],
                        [2.0 * z, 2.0 * w, -2.0 * x]])

    dQdy = numpy.array([[-2 * y, 2 * x, 2 * w],
                        [2 * x, 2 * y, 2 * z],
                        [-2 * w, 2 * z, -2 * y]])

    dQdz = numpy.array([[-2 * z, -2 * w, 2 * x],
                        [2 * w, -2 * z, 2 * y],
                        [2 * x, 2 * y, 2 * z]])

    Cv = numpy.zeros((3, 3, 3, 3))

    Cv = Cvoigt(C)

    Crot = numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, Q)

    dCrotdw = numpy.einsum('ip, jq, pqrs, kr, ls', dQdw, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdw, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdw, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdw)

    dCrotdx = numpy.einsum('ip, jq, pqrs, kr, ls', dQdx, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdx, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdx, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdx)

    dCrotdy = numpy.einsum('ip, jq, pqrs, kr, ls', dQdy, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdy, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdy, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdy)

    dCrotdz = numpy.einsum('ip, jq, pqrs, kr, ls', dQdz, Q, Cv, Q, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, dQdz, Cv, Q, Q) + \
        numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, dQdz, Q) + numpy.einsum('ip, jq, pqrs, kr, ls', Q, Q, Cv, Q, dQdz)

    return Crot, dCrotdw, dCrotdx, dCrotdy, dCrotdz, Q

cpdef inline numpy.ndarray[numpy.double_t, ndim = 4] Cvoigt(numpy.ndarray[numpy.double_t, ndim = 2] Ch):
    cdef numpy.ndarray[numpy.double_t, ndim = 4] C
    cdef int i, j, k, l, n, m

    C = numpy.zeros((3, 3, 3, 3))

    voigt = [[(0, 0)], [(1, 1)], [(2, 2)], [(1, 2), (2, 1)], [(0, 2), (2, 0)], [(0, 1), (1, 0)]]

    for i in range(6):
        for j in range(6):
            for k, l in voigt[i]:
                for n, m in voigt[j]:
                    C[k, l, n, m] = Ch[i, j]
    return C

cpdef buildKM(numpy.ndarray[numpy.double_t, ndim = 2] Ch, numpy.ndarray[numpy.double_t, ndim = 4] dp, numpy.ndarray[numpy.double_t, ndim = 2] pv, double density):
    cdef numpy.ndarray[numpy.double_t, ndim = 6] dpe
    cdef numpy.ndarray[numpy.double_t, ndim = 4] C, Kt, Mt
    cdef numpy.ndarray[numpy.double_t, ndim = 2] K, M
    cdef int i, j, k, l, n, m, N

    N = dp.shape[0]

    C = numpy.zeros((3, 3, 3, 3))

    voigt = [[(0, 0)], [(1, 1)], [(2, 2)], [(1, 2), (2, 1)], [(0, 2), (2, 0)], [(0, 1), (1, 0)]]

    for i in range(6):
        for j in range(6):
            for k, l in voigt[i]:
                for n, m in voigt[j]:
                    C[k, l, n, m] = Ch[i, j]

    Kt = numpy.zeros((N, 3, N, 3))

    for n in range(N):
        for m in range(N):
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        for l in range(3):
                            Kt[n, i, m, k] += C[i, j, k, l] * dp[n, m, j, l]

    K = Kt.reshape((N * 3, N * 3))

    Mt = numpy.zeros((dp.shape[0], 3, dp.shape[0], 3))
    for n in range(N):
        for m in range(N):
            Mt[n, 0, m, 0] = density * pv[n, m]
            Mt[n, 1, m, 1] = density * pv[n, m]
            Mt[n, 2, m, 2] = density * pv[n, m]

    M = Mt.reshape(3 * pv.shape[0], 3 * pv.shape[0])

    return K, M

#%%
#
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
# DON'T GO BELOW HERE
#
#
#reload(polybasisqu)
#
#dp1, pv1, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)
#
#for i in range(3):
#    factors = numpy.array([X, Y, Z])
#    factors[i] *= 1.0001
#    Xt, Yt, Zt = factors
#    dp2, pv2, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, Xt, Yt, Zt)
#
#    ddps = [ddpdX, ddpdY, ddpdZ]
#    dpvs = [dpvdX, dpvdY, dpvdZ]
#
#    print (dp2[1, 1] - dp1[1, 1]) / (factors[i] - factors[i] / 1.0001)
#    print ddps[i][1, 1]
#    print (pv2[0, 0] - pv1[0, 0]) / (factors[i] - factors[i] / 1.0001)
#    print dpvs[i][0, 0]
#    print '----'
#
#dp1, pv1, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)
#
#C = numpy.array([[c11, c12, c12, 0, 0, 0],
#                     [c12, c11, c12, 0, 0, 0],
#                     [c12, c12, c11, 0, 0, 0],
#                     [0, 0, 0, c44, 0, 0],
#                     [0, 0, 0, 0, c44, 0],
#                     [0, 0, 0, 0, 0, c44]])
#
#C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, w, x, y, z)
#
#K1, M1 = polybasisqu.buildKM(C, dp1, pv1, density)
#
#a, b, y = 0.1, 0.2, 0.3
#
#for i in range(3):
#    factors = numpy.array([X, Y, Z])
#    factors[i] *= 1.0001
#    Xt, Yt, Zt = factors
#    dp2, pv2, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, Xt, Yt, Zt)
#
#    dKdX, dMdX = polybasisqu.buildKM(C, ddpdX, dpvdX, density)
#    dKdY, dMdY = polybasisqu.buildKM(C, ddpdY, dpvdY, density)
#    dKdZ, dMdZ = polybasisqu.buildKM(C, ddpdZ, dpvdZ, density)
#
#    K2, M2 = polybasisqu.buildKM(C, dp2, pv2, density)
#
#    ddps = [dKdX, dKdY, dKdZ]
#    dpvs = [dMdX, dMdY, dMdZ]
#
#    print (K2[494, 486] - K1[494, 486]) / (factors[i] - factors[i] / 1.0001)
#    print ddps[i][494, 486]
#    print (M2[0, 12] - M1[0, 12]) / (factors[i] - factors[i] / 1.0001)
#    print dpvs[i][0, 12]
#    print '----'
##%%
#reload(polybasisqu)
#
#dp, pv, ddpdX, ddpdY, ddpdZ, dpvdX, dpvdY, dpvdZ = polybasisqu.build(N, X, Y, Z)
#
#C = numpy.array([[c11, c12, c12, 0, 0, 0],
#                     [c12, c11, c12, 0, 0, 0],
#                     [c12, c12, c11, 0, 0, 0],
#                     [0, 0, 0, c44, 0, 0],
#                     [0, 0, 0, 0, c44, 0],
#                     [0, 0, 0, 0, 0, c44]])
#
#C1, dCdw1, dCdx1, dCdy1, dCdz1, K = polybasisqu.buildRot(C, w, x, y, z)
#
#dCdc11 = K.dot(numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))
#
#dCdc12 = K.dot(numpy.array([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#                                [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                                [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))
#
#dCdc44 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).dot(K.T))
#
#    #tmp = time.time()
#dKdw1, _ = polybasisqu.buildKM(dCdw1, dp, pv, density)
#dKdx1, _ = polybasisqu.buildKM(dCdx1, dp, pv, density)
#dKdy1, _ = polybasisqu.buildKM(dCdy1, dp, pv, density)
#dKdz1, _ = polybasisqu.buildKM(dCdz1, dp, pv, density)
#
#dKdc111, _ = polybasisqu.buildKM(dCdc11, dp, pv, density)
#dKdc121, _ = polybasisqu.buildKM(dCdc12, dp, pv, density)
#dKdc441, _ = polybasisqu.buildKM(dCdc44, dp, pv, density)
#
#dKdX1, dMdX1 = polybasisqu.buildKM(C1, ddpdX, dpvdX, density)
#dKdY1, dMdY1 = polybasisqu.buildKM(C1, ddpdY, dpvdY, density)
#dKdZ1, dMdZ1 = polybasisqu.buildKM(C1, ddpdZ, dpvdZ, density)
#
#K1, M1 = polybasisqu.buildKM(C1, dp, pv, density)
#
#w = 0.5
#x = 0.3
#y = 0.1
#z = numpy.sqrt(1 - w**2 - x**2 - y**2)
#
#for i in range(4):
#    #factors = numpy.array([c11, c12, c44])
#    factors = numpy.array([w, x, y, z])
#    factors[i] *= 1.0001
#    #c11t, c12t, c44t = factors
#    wt, xt, yt, zt = factors
#    c11t, c12t, c44t = c11, c12, c44
#
#    C = numpy.array([[c11t, c12t, c12t, 0, 0, 0],
#                     [c12t, c11t, c12t, 0, 0, 0],
#                     [c12t, c12t, c11t, 0, 0, 0],
#                     [0, 0, 0, c44t, 0, 0],
#                     [0, 0, 0, 0, c44t, 0],
#                     [0, 0, 0, 0, 0, c44t]])
#
#    C, dCdw, dCdx, dCdy, dCdz, K = polybasisqu.buildRot(C, wt, xt, yt, zt)
#
#    dCdc11 = K.dot(numpy.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))
#
#    dCdc12 = K.dot(numpy.array([[0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
#                                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
#                                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).dot(K.T))
#
#    dCdc44 = K.dot(numpy.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
#                                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).dot(K.T))
#
#        #tmp = time.time()
#    dKdw, _ = polybasisqu.buildKM(dCdw, dp, pv, density)
#    dKdx, _ = polybasisqu.buildKM(dCdx, dp, pv, density)
#    dKdy, _ = polybasisqu.buildKM(dCdy, dp, pv, density)
#    dKdz, _ = polybasisqu.buildKM(dCdz, dp, pv, density)
#
#    dKdc11, _ = polybasisqu.buildKM(dCdc11, dp, pv, density)
#    dKdc12, _ = polybasisqu.buildKM(dCdc12, dp, pv, density)
#    dKdc44, _ = polybasisqu.buildKM(dCdc44, dp, pv, density)
#
#    dKdX, dMdX = polybasisqu.buildKM(C, ddpdX, dpvdX, density)
#    dKdY, dMdY = polybasisqu.buildKM(C, ddpdY, dpvdY, density)
#    dKdZ, dMdZ = polybasisqu.buildKM(C, ddpdZ, dpvdZ, density)
#
#    K, M = polybasisqu.buildKM(C, dp, pv, density)
#
#    #ders = [dCdw, dCdx, dCdy, dCdz]
#    ders = [dKdw, dKdx, dKdy, dKdz]
#    #ders = [dKdc111, dKdc121, dKdc441]
#
#    print ((K - K1) / (factors[i] - factors[i] / 1.0001))
#    print ders[i]
#    print '--'
