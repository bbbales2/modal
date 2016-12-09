#%%

import numpy
import pyximport
pyximport.install()#reload_support = True)
import polybasisqu
import scipy.linalg
import numbers
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except:
    print "import matplotlib.pyplot as plt failed. No plotting available through rus library"
    matplotlib_available = False

import bisect
#%%
import time
#%%

import contextlib

# Stolen from http://stackoverflow.com/a/2891805/3769360
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    original = numpy.get_printoptions()
    numpy.set_printoptions(*args, **kwargs)
    yield
    numpy.set_printoptions(**original)

class HMC():
    def __init__(self, density, X, Y, Z, resonance_modes, stiffness_matrix, parameters, constrained_positive = None, rotations = None, T = 1.0, tol = 1e-3, maxN = 12, N = None):
        self.C = stiffness_matrix
        self.dC = {}
        self.density = density
        self.X = X
        self.Y = Y
        self.Z = Z
        self.T = T
        self.order = {}
        self.initial_conditions = parameters
        self.labels = {}
        self.maxN = maxN

        if N is not None:
            print "WARNING: Parameter 'N' to HMC.__init__ is defunct. It is ignored in favor of maxN"

        if isinstance(resonance_modes[0], numbers.Number):
            resonance_modes = [resonance_modes]

        self.rotations = rotations

        if type(self.rotations) == bool:
            if self.rotations:
                self.rotations = [0] * len(resonance_modes)
            else:
                self.rotations = None

        if self.rotations:
            self.R = max(self.rotations) + 1
        else:
            self.R = 0

        self.constrained_positive = set()
        if constrained_positive != None:
            self.constrained_positive = set(constrained_positive)

        self.constrained_positive.add('std')

        # Choose an ordering of internal parameters
        for i, p in enumerate(self.initial_conditions):
            self.order[p] = i
            self.labels[p] = str(i)

        self.labels['std'] = 'std'

        self.data = resonance_modes
        self.S = len(resonance_modes)

        self.modes = []
        for s in range(self.S):
            self.modes.append(len(self.data[s]))

        for p in self.initial_conditions:
            if p != 'std':
                self.dC[p] = self.C.diff(p)

        self.reset(self.initial_conditions)

        self.dps = {}
        self.pvs = {}

        self.compute_resolutions(tol) # This will fill up self.dps and self.pvs

        self.set_resolution()

    def reset(self, initial_conditions = None):
        if initial_conditions == None:
            initial_conditions = self.initial_conditions
        else:
            self.initial_conditions = initial_conditions

        self.qs = []
        self.qrs = []
        self.logps = []
        self.accepts = []

        self.current_q = numpy.zeros(len(self.order))

        self.current_qr = numpy.zeros((self.R, 4))

        for r in range(self.R):
            self.current_qr[r, 0] = 1.0
            #self.current_qr[r, :] = numpy.random.randn(4)
            #self.current_qr[r, :] /= numpy.linalg.norm(self.current_qr[r, :])

        for p, i in self.order.items():
            self.current_q[i] = numpy.log(initial_conditions[p]) if p in self.constrained_positive else initial_conditions[p]

        self.accepts.append(self.current_q)

    # Building a lookup that returns, given a number of modes, the order of the polynomials N in the Rayleigh Ritz expansion
    def compute_resolutions(self, tol):
        # This will hold result of computing 2D array of all freqs
        freqs = {}

        Ns = range(8, self.maxN + 4, 2)

        for N in Ns:
            qdict = self.qdict(self.current_q)
            qr = self.current_qr

            self.dps[N] = {}
            self.pvs[N] = {}

            freqsTmp = []

            for p in qdict:
                qdict[p] = numpy.exp(qdict[p]) if p in self.constrained_positive else qdict[p]

            for s in range(self.S):
                C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

                if self.rotations:
                    w, x, y, z = qr[self.rotations[s]]

                    C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

                dp, pv, _, _, _, _, _, _ = polybasisqu.build(N, self.X[s], self.Y[s], self.Z[s])

                self.dps[N][s] = dp
                self.pvs[N][s] = pv

                K, M = polybasisqu.buildKM(C, dp, pv, self.density[s])

                eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + max(self.modes) - 1))

                freqsTmp.append(numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000))

            freqs[N] = freqsTmp

        resolutions = []
        Nvs = []

        for N in Ns[:-1]:
            # Compute for each N the maximum number of modes it gets accurate vs. the highest order approx.
            lt = max(self.modes) + 1
            for s in range(self.S):
                errors = numpy.abs((freqs[N][s] - freqs[Ns[-1]][s]) / freqs[Ns[-1]][s])

                for i in range(len(errors)):
                    if errors[i] >= tol:
                        lt = min(lt, i)
                        break

            if lt not in resolutions and lt > 0:
                resolutions.append(lt - 1)
                Nvs.append(N)

        if len(Nvs) > 0:
            supremumN = min(set(Ns) - set(range(max(Nvs) + 1)))
        else:
            supremumN = Ns[0]

        # Set for each number of modes the lowest N which gets things right (within tol)
        self.resolutions = {}
        for m in range(max(self.modes) + 1):
            i = bisect.bisect_left(resolutions, m)

            if i < len(Nvs):
                self.resolutions[m] = Nvs[i]
            else:
                self.resolutions[m] = supremumN

        return self.resolutions

    def set_resolution(self, number_of_modes = -1):

        if number_of_modes == -1:
            N = self.resolutions[max(self.modes)]

            self.dp = self.dps[N]
            self.pv = self.pvs[N]

            self.resolution = max(self.modes)
        else:
            N = self.resolutions[number_of_modes]

            self.dp = self.dps[N]
            self.pv = self.pvs[N]

            self.resolution = number_of_modes

        return N

    def qdict(self, q):
        result = {}

        for p, i in self.order.items():
            result[p] = q[i]

        return result

    def UgradU(self, q, qr):
        qdict = self.qdict(q)

        for p in qdict:
            if p in self.constrained_positive:
                qdict[p] = numpy.exp(qdict[p])

        std = qdict['std']

        C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

        logp_total = 0.0
        dlogpdq_total = numpy.zeros(len(self.order))

        if self.rotations:
            dlogpdqr_total = numpy.zeros((self.R, 4))

        freqs = {}
        dfreqsdl = {}

        dldps = {}
        if self.rotations:
            for s, r in enumerate(self.rotations):
                w, x, y, z = qr[r]

                Cr, dCdw, dCdx, dCdy, dCdz, Kr = polybasisqu.buildRot(C, w, x, y, z)

                dKdws, _ = polybasisqu.buildKM(dCdw, self.dp[s], self.pv[s], self.density[s])
                dKdxs, _ = polybasisqu.buildKM(dCdx, self.dp[s], self.pv[s], self.density[s])
                dKdys, _ = polybasisqu.buildKM(dCdy, self.dp[s], self.pv[s], self.density[s])
                dKdzs, _ = polybasisqu.buildKM(dCdz, self.dp[s], self.pv[s], self.density[s])

                # Likelihood p(data | params)
                K, M = polybasisqu.buildKM(Cr, self.dp[s], self.pv[s], self.density[s])

                eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + self.resolution - 1))#max(self.modes)

                freqs[r] = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)
                dfreqsdl[r] = 0.5e11 / (numpy.sqrt(eigs * 1e11) * numpy.pi * 2000)

                for p, i in self.order.items():
                    if p == 'std':
                        continue

                    dKdp, _ = polybasisqu.buildKM(Kr.dot(numpy.array(self.dC[p].evalf(subs = qdict)).astype('float')).dot(Kr.T), self.dp[s], self.pv[s], self.density[s])

                    dldps[(p, r)] = numpy.array([evecs[:, j].T.dot(dKdp.dot(evecs[:, j])) for j in range(evecs.shape[1])])

                dldps[("w", r)] = numpy.array([evecs[:, j].T.dot(dKdws.dot(evecs[:, j])) for j in range(evecs.shape[1])])
                dldps[("x", r)] = numpy.array([evecs[:, j].T.dot(dKdxs.dot(evecs[:, j])) for j in range(evecs.shape[1])])
                dldps[("y", r)] = numpy.array([evecs[:, j].T.dot(dKdys.dot(evecs[:, j])) for j in range(evecs.shape[1])])
                dldps[("z", r)] = numpy.array([evecs[:, j].T.dot(dKdzs.dot(evecs[:, j])) for j in range(evecs.shape[1])])
        else:
            K, M = polybasisqu.buildKM(C, self.dp[0], self.pv[0], self.density[0])

            eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + self.resolution - 1))#max(self.modes)

            freqs[0] = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)
            dfreqsdl[0] = 0.5e11 / (numpy.sqrt(eigs * 1e11) * numpy.pi * 2000)

            for p, i in self.order.items():
                if p == 'std':
                    continue

                dKdp, _ = polybasisqu.buildKM(numpy.array(self.dC[p].evalf(subs = qdict)).astype('float'), self.dp[s], self.pv[s], self.density[s])

                dldps[(p, 0)] = numpy.array([evecs[:, j].T.dot(dKdp.dot(evecs[:, j])) for j in range(evecs.shape[1])])

        for s in range(self.S):
            if self.rotations:
                r = self.rotations[s]
            else:
                r = 0

            dlpdfreqs = numpy.zeros(min(self.resolution, self.modes[s]))
            dlpdstd = 0.0
            logp = 0.0

            for i in range(min(self.resolution, self.modes[s])):
                dlpdfreqs[i] = (self.data[s][i] - freqs[r][i]) / (std ** 2)
                dlpdstd += ((-(std ** 2) + (self.data[s][i] - freqs[r][i]) ** 2) / (std ** 3))
                logp += (0.5 * (-((self.data[s][i] - freqs[r][i]) **2 / (std**2)) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

            #for i in range(self.modes[s]):
            #    dlpdfreqs[i] = (self.data[s][i] - freqs[r][i]) / ((i + 1)**2 * std ** 2)
            #    dlpdstd += ((-((i + 1)**2 * std ** 2) + (self.data[s][i] - freqs[r][i]) ** 2) / ((i + 1)**2 * std ** 3))
            #    logp += (0.5 * (-((self.data[s][i] - freqs[r][i]) **2 / ((i + 1)**2 * std**2)) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(std)))

            dlpdstd = dlpdstd * qdict['std'] + 1
            logp_total += logp + sum([q[self.order[p]] for p in self.constrained_positive])

            dlpdl = dlpdfreqs * dfreqsdl[r]

            for p, i in self.order.items():
                if p == 'std':
                    continue

                dlpdp = dlpdl.dot(dldps[(p, r)])

                dlogpdq_total[i] += (dlpdp * qdict[p] + 1) if p in self.constrained_positive else dlpdp

            if self.rotations:
                dlogpdqr_total[r, 0] += dlpdl.dot(dldps[("w", r)])
                dlogpdqr_total[r, 1] += dlpdl.dot(dldps[("x", r)])
                dlogpdqr_total[r, 2] += dlpdl.dot(dldps[("y", r)])
                dlogpdqr_total[r, 3] += dlpdl.dot(dldps[("z", r)])

            dlogpdq_total[self.order['std']] += dlpdstd

        return -logp_total / self.T, -dlogpdq_total / self.T, -dlogpdqr_total / self.T if self.rotations else None

    def set_labels(self, labels):
        self.labels = labels

    def set_timestepping(self, epsilon = 0.0001, L = 50, param_scaling = None):
        self.epsilon = epsilon

        if param_scaling != None:
            self.epsilon = numpy.ones(len(self.order)) * self.epsilon

            for p in param_scaling:
                self.epsilon[self.order[p]] *= param_scaling[p]

        self.L = L

    def sample(self, steps = -1, debug = False, silent = False):
        if not hasattr(self, 'epsilon'):
            raise Exception("Must call 'rus.HMC.set_timestepping' before 'rus.HMC.sample'")

        epsilon = self.epsilon
        L = self.L

        step = 0
        while step < steps or steps == -1:
            q = self.current_q.copy()
            qr = self.current_qr.copy()

            p = numpy.random.randn(len(q)) # independent standard normal variates
            pr = numpy.random.randn(*qr.shape)

            for r in range(self.R):
                pr[r] -= numpy.outer(qr[r], qr[r]).dot(pr[r])

            current_p = p.copy()
            current_pr = pr.copy()
            # Make a half step for momentum at the beginning
            U, gradU, gradUr = self.UgradU(q, qr)
            p = p - epsilon * gradU / 2

            if self.rotations:
                pr = pr - epsilon * gradUr / 2

                for r in range(self.R):
                    pr[r] -= numpy.outer(qr[r], qr[r]).dot(pr[r])

            # Alternate full steps for position and momentum
            for i in range(L):
                # Make a full step for the position
                q = q + epsilon * p

                for r in range(self.R):
                    alpha = numpy.linalg.norm(pr[r])

                    m1 = numpy.array([[1.0, 0.0],
                                      [0.0, 1 / alpha]])

                    m2 = numpy.array([[numpy.cos(alpha * epsilon), -numpy.sin(alpha * epsilon)],
                                      [numpy.sin(alpha * epsilon), numpy.cos(alpha * epsilon)]])

                    m3 = numpy.array([[1.0, 0.0],
                                      [0.0, alpha]])

                    xv = numpy.array([qr[r], pr[r]]).T.dot(m1.dot(m2.dot(m3)))

                    qr[r] = xv[:, 0]
                    pr[r] = xv[:, 1]

                    qr[r] /= numpy.linalg.norm(qr[r])

                #q[-3:] = inv_rotations.qu2eu(symmetry.Symmetry.Cubic.fzQuat(quaternion.Quaternion(inv_rotations.eu2qu(q[-3:]))))
                # Make a full step for the momentum, except at end of trajectory
                if i != L - 1:
                    U, gradU, gradUr = self.UgradU(q, qr)
                    p = p - epsilon * gradU

                    if self.rotations:
                        pr = pr - epsilon * gradUr

                        for r in range(self.R):
                            pr[r] -= numpy.outer(qr[r], qr[r]).dot(pr[r])

                if debug:
                    print "New q: {0}".format(self.print_q(q, qr))
                    print "H (constant or decreasing): ", U + sum(p ** 2) / 2, U, sum(p **2) / 2.0
                    print ""

            U, gradU, gradUr = self.UgradU(q, qr)
            # Make a half step for momentum at the end.
            p = p - epsilon * gradU / 2

            if self.rotations:
                pr = pr - epsilon * gradUr / 2

                for s in range(self.R):
                    pr[r] -= numpy.outer(qr[r], qr[r]).dot(pr[r])

            # Negate momentum at end of trajectory to make the proposal symmetric
            p = -p
            pr = -pr
            # Evaluate potential and kinetic energies at start and end of trajectory
            UC, _, _ = self.UgradU(self.current_q, self.current_qr)
            current_U = UC
            current_K = sum(current_p ** 2) / 2 + (current_pr ** 2).sum() / 2
            proposed_U = U
            proposed_K = sum(p ** 2) / 2 + (pr ** 2).sum() / 2

            # Accept or reject the state at end of trajectory, returning either
            # the position at the end of the trajectory or the initial position
            dQ = current_U - proposed_U + current_K - proposed_K

            with printoptions(precision = 5):
                if numpy.random.rand() < min(1.0, numpy.exp(dQ)):
                    self.current_q = q # accept
                    self.current_qr = qr

                    self.accepts.append(len(self.qs) - 1)

                    self.logps.append(U)

                    if not silent or debug:
                        print "Accepted ({0} accepts so far):".format(len(self.accepts))
                        print self.print_q(self.current_q, self.current_qr)
                else:
                    if not silent or debug:
                        print "Rejected: "
                        print self.print_q(self.current_q, self.current_qr)

                    self.logps.append(UC)

                self.qs.append(self.current_q.copy())
                self.qrs.append(self.current_qr.copy())

                if not silent or debug:
                    print "Energy change ({0} samples, {1} accepts): ".format(len(self.qs), len(self.accepts)), min(1.0, numpy.exp(dQ)), dQ, current_U, proposed_U, current_K, proposed_K

            step += 1

    def print_q(self, q = None, qr = None, precision = 5):
        if q is None:
            q = self.current_q

        if qr is None:
            qr = self.current_qr

        out = []

        llist = []
        for p, name in self.labels.items():
            if p in self.order:
                llist.append("{0} : {1:0.{2}f}".format(name, numpy.exp(q[self.order[p]]) if p in self.constrained_positive else q[self.order[p]], precision))

        out.append("{{ {0} }}".format(", ".join(llist)))

        if self.rotations:
            for r in range(self.R):
                llist = []
                for name, i in zip(["w", "x", "y", "z"], range(4)):
                    llist.append("{0} : {1:0.{2}f}".format(name, qr[r, i], precision))

                out.append("rotation {0} : {{ {1} }}".format(r, ", ".join(llist)))

        return "\n".join(out)

    def print_current(self, precision = 5):
        qdict = self.qdict(self.current_q)
        qr = self.current_qr

        for p in qdict:
            qdict[p] = numpy.exp(qdict[p]) if p in self.constrained_positive else qdict[p]

        for s in range(self.S):
            C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

            if self.rotations:
                w, x, y, z = qr[self.rotations[s]]

                C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

            K, M = polybasisqu.buildKM(C, self.dp[s], self.pv[s], self.density[s])

            eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + max(self.modes) - 1))

            freqs = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)

            errors = freqs - self.data[s]

            print "Sample: {0}".format(s)
            if self.rotations:
                print "Rotation: {0}".format(self.rotations[s])
            print "Mean error: ", numpy.mean(errors)
            print "Std deviation in error: ", numpy.std(errors)

            print "Computed, Measured"
            for i, freq in enumerate(freqs):
                if i < len(self.data[s]):
                    print "{0:0.{2}f}, {1:0.{2}f}".format(freq, self.data[s][i], precision)

    def posterior_predictive(self, lastN = 200, precision = 5, plot = True, which_samples = None):
        lastN = min(lastN, len(self.qs))

        posterior_predictive = numpy.zeros((max(self.modes), lastN, self.S))

        if which_samples == None:
            which_samples = range(self.S)

        for i, (q, qr) in enumerate(zip(self.qs[-lastN:], self.qrs[-lastN:])):
            for s in which_samples:#range(self.S):
                qdict = self.qdict(q)

                for p in qdict:
                    qdict[p] = numpy.exp(qdict[p]) if p in self.constrained_positive else qdict[p]

                C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

                if self.rotations:
                    w, x, y, z = qr[self.rotations[s]]

                    C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

                K, M = polybasisqu.buildKM(C, self.dp[s], self.pv[s], self.density[s])

                eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + max(self.modes) - 1))

                posterior_predictive[:, i, s] = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)
        #print l, r, posterior_predictive[0]

        for s in which_samples:#range(self.S):
            if self.rotations:
                r = self.rotations[s]
            else:
                r = 0

            ppl = numpy.percentile(posterior_predictive[:, :, s], 2.5, axis = 1)
            ppr = numpy.percentile(posterior_predictive[:, :, s], 97.5, axis = 1)

            if plot and matplotlib_available:
                data = []

                for l in range(len(self.data[s])):
                    tmp = []
                    for ln in range(lastN):
                        tmp.append(posterior_predictive[l, ln, s] - self.data[s][l])

                    data.append(tmp)

                data = numpy.array(data)
                plt.boxplot(numpy.array(data).transpose())

                #ax1 = plt.gca()

                #for ll, meas, rr, tick in zip(ppl, self.data[s], ppr, range(len(self.data[s]))):
                #    ax1.text(tick + 1, ax1.get_ylim()[1] * 0.90, '{0:10.{3}f} {1:10.{3}f} {2:10.{3}f}'.format(ll, meas, rr, precision),
                #             horizontalalignment='center', rotation=45, size='x-small')
                plt.xlabel('Mode')
                plt.ylabel('Computed - Measured (khz)')
            else:
                print "For dataset {0}".format(s)
                print "{0:8s} {1:10s} {2:10s} {3:10s}".format("Outside", "2.5th %", "measured", "97.5th %")
                for ll, meas, rr in zip(ppl, self.data[s], ppr):
                    print "{0:8s} {1:10.{4}f} {2:10.{4}f} {3:10.{4}f}".format("*" if (meas < ll or meas > rr) else " ", ll, meas, rr, precision)

    def save(self, filename):
        f = open(filename, 'w')
        f.write(self.saves())
        f.close()

    def format_samples(self, lastN = -1):
        header = []
        samples = []

        printOrder = []
        do_exp = []

        if lastN == -1:
            qs = self.qs
            qrs = self.qrs
        else:
            qs = self.qs[-lastN:]
            qrs = self.qrs[-lastN:]

        for p, name in self.labels.items():
            if p in self.order:
                header.append(name)
                printOrder.append(self.order[p])

                if p in self.constrained_positive:
                    do_exp.append(True)
                else:
                    do_exp.append(False)

        for i, do_exp_ in zip(printOrder, do_exp):
            tmp_samples = []

            for q in qs:
                tmp_samples.append(numpy.exp(q[i]) if do_exp_ else q[i])

            samples.append(tmp_samples)

        for r in range(self.R):
            for name, i in zip(["w", "x", "y", "z"], range(4)):

                header.append("{0}_{1}".format(name, r))

                tmp_samples = []
                for qr in qrs:
                    tmp_samples.append(qr[r, i])

                samples.append(tmp_samples)

        return header, samples

    def saves(self):
        output = []

        header, samples = self.format_samples()

        output.append("# {0}".format(", ".join(header)))

        for params in zip(*samples):
            output.append(", ".join([str(sample) for sample in params]))

        return "\n".join(output)

    def derivative_check(self):
        for p, i in self.order.items():
            name = self.labels[p]

            q = self.current_q.copy()

            q2 = q.copy()

            if numpy.abs(q2[i] - q[i]) < 1.0:
                q2[i] = q[i] + 1e-7
            else:
                q2[i] *= 1.0000001

            U1, gradU, _ = self.UgradU(q, self.current_qr)
            U2, _, _ = self.UgradU(q2, self.current_qr)

            print "dlogpd{{{0}}}".format(name)
            print "   Computed: ", (U2 - U1) / (q2[i] - q[i])
            print "   Analytical: ", gradU[i]

        if self.rotations:
            for r in range(self.R):
                for i, name in zip(range(4), ["w", "x", "y", "z"]):
                    qr = self.current_qr.copy()

                    qr2 = qr.copy()

                    if numpy.abs(qr2[r, i] - qr[r, i]) < 1.0:
                        qr2[r, i] = qr[r, i] + 1e-5
                    else:
                        qr2[r, i] *= 1.0000001

                    U1, _, gradUr = self.UgradU(self.current_q, qr)
                    U2, _, _ = self.UgradU(self.current_q, qr2)

                    print "dlogpd{{{0}}}-{{{1}}}".format(name, r)
                    print "   Computed: ", (U2 - U1) / (qr2[r, i] - qr[r, i])
                    print "   Analytical: ", gradUr[r, i]
