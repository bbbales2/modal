#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy
import time

cpdef inline float polyint(int n, int m, int l):
    cdef float xtmp, ytmp

    if n < 0 or m < 0 or l < 0 or n % 2 > 0 or m % 2 > 0 or l % 2 > 0:
        return 0.0

    xtmp = 2 * 0.5**(n + 1)
    ytmp = 2 * 0.5**(m + 1) * xtmp
    return 2 * 0.5**(l + 1) * ytmp / ((n + 1) * (m + 1) * (l + 1))

cpdef build(int N, float X, float Y, float Z):
    cdef numpy.ndarray[numpy.float_t, ndim = 4] dp, ddpdX, ddpdY, ddpdZ
    cdef numpy.ndarray[numpy.float_t, ndim = 2] pv, dpvdX, dpvdY, dpvdZ
    cdef numpy.ndarray[numpy.int_t, ndim = 2] basis
    cdef numpy.ndarray[numpy.float_t, ndim = 1] Xs, Ys, Zs

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

cpdef buildKM(numpy.ndarray[numpy.float_t, ndim = 2] Ch, numpy.ndarray[numpy.float_t, ndim = 4] dp, numpy.ndarray[numpy.float_t, ndim = 2] pv, float density):
    cdef numpy.ndarray[numpy.float_t, ndim = 6] dpe
    cdef numpy.ndarray[numpy.float_t, ndim = 4] C, Kt, Mt
    cdef numpy.ndarray[numpy.float_t, ndim = 2] K, M
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