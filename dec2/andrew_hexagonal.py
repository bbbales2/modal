#%%

import numpy
import time
import scipy
import sympy
import os
os.chdir('/home/bbales2/modal')

import rus
reload(rus)
#%%
#656 samples
#%%

## Dimensions for TF-2
X = 0.0055#0.007753
Y = 0.01079#0.009057
Z = 0.013025#0.013199

#Sample density
density = 8114.300888#4401.695921 #Ti-64-TF2

# Ti-64-TF2 Test Data
data = numpy.array([84.824,
104.067,
121.173,
131.982,
146.605,
159.923,
170.609,
177.603,
179.117,
192.316,
213.764,
214.697,
219.536,
220.501,
236.464,
256.867,
262.779,
267.559,
270.429,
280.313,
303.722,
309.733,
313.656,
319.363,
326.227,
331.307,
334.499,
337.853,
338.842,
341.585])
#%%

# These are the two HMC parameters
#   L is the number of timesteps to take -- use this if samples in the traceplots don't look random
#   epsilon is the timestep -- make this small enough so that pretty much all the samples are being accepted, but you
#       want it large enough that you can keep L ~ 50 -> 100 and still get independent samples
L = 50
# start epsilon at .0001 and try larger values like .0005 after running for a while
# epsilon is timestep, we want to make as large as possibe, wihtout getting too many rejects
epsilon = 0.0001

# Set this to true to debug the L and eps values
debug = False

#%%

reload(rus)

c11, c12, c13, c33, c44 = sympy.symbols('c11 c12 c13 c33 c44')

C = sympy.Matrix([[c11, c12, c13, 0, 0, 0],
                     [c12, c11, c13, 0, 0, 0],
                     [c13, c13, c33, 0, 0, 0],
                     [0, 0, 0, c44, 0, 0],
                     [0, 0, 0, 0, c44, 0],
                     [0, 0, 0, 0, 0, (c11 - c12) / 2.0]])

hmc = rus.HMC(tol = 1e-3,
              density = density, X = X, Y = Y, Z = Z,
              resonance_modes = data, # List of resonance modes
              stiffness_matrix = C, # Stiffness matrix
              parameters = { c11 : 2.0, c12 : 1.0, c13 : 1.0, c33 : 1.0, c44 : 1.0, 'std' : 5.0 }, # Parameters
              constrained_positive = [c11, c12, c13, c33, c44],
              rotations = True,
              T = 1.0)

hmc.set_labels({ c11 : 'c11', c12 : 'c12', c13 : 'c13', c33 : 'c33', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)
hmc.sample(steps = 5, debug = True)
#%%
hmc.set_timestepping(epsilon = epsilon * 10.0, L = 50)
hmc.sample(debug = False)#True)#False)#True)
#%%
hmc.derivative_check()
#%%
hmc.set_timestepping(epsilon = epsilon * 20, L = 75)
hmc.sample(debug = True)#False)#True)
#%%
hmc.print_current()
#%%
hmc.posterior_predictive(plot = True, lastN = 200)
plt.title('Posterior predictive', fontsize = 72)
plt.xlabel('Mode', fontsize = 48)
plt.ylabel('Computed - Measured (khz)', fontsize = 48)
plt.tick_params(axis='y', which='major', labelsize=48)
plt.tick_params(axis='x', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches((24, 16))
plt.savefig('dec2/cmsxhigh/posteriorpredictive.png', dpi = 144)
plt.show()
#%%
hmc.save('/home/bbales2/modal/paper/cmsx4/qs.csv')
#%%
import pickle

f = open('/home/bbales2/modal/dec2/cmsx4high.pkl', 'w')
pickle.dump(hmc, f)
f.close()
#%%
import polybasisqu
import pandas
import seaborn
import matplotlib.pyplot as plt

def posterior_predictive(self, lastN = 200, precision = 5, plot = True):
        lastN = min(lastN, len(self.qs))

        posterior_predictive = numpy.zeros((max(self.modes), lastN, max(self.R, 1)))

        for i, (q, qr) in enumerate(zip(self.qs[-lastN:], self.qrs[-lastN:])):
            for r in range(max(self.R, 1)):
                qdict = self.qdict(q)

                for p in qdict:
                    qdict[p] = numpy.exp(qdict[p]) if p in self.constrained_positive else qdict[p]

                C = numpy.array(self.C.evalf(subs = qdict)).astype('float')

                if self.rotations:
                    w, x, y, z = qr[self.rotations[r]]

                    C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

                K, M = polybasisqu.buildKM(C, self.dp, self.pv, self.density)

                eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + max(self.modes) - 1))

                posterior_predictive[:, i, r] = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000)
        #print l, r, posterior_predictive[0]

        for s in range(self.S):
            if self.rotations:
                r = self.rotations[s]
            else:
                r = 0

            ppl = numpy.percentile(posterior_predictive[:, :, r], 2.5, axis = 1)
            ppr = numpy.percentile(posterior_predictive[:, :, r], 97.5, axis = 1)

            if plot:
                data = []

                for l in range(len(self.data[s])):
                    tmp = []
                    for ln in range(lastN):
                        tmp.append(posterior_predictive[l, ln, r] - self.data[s][l])

                    data.append(tmp)
                    #data.append([l, self.data[s][l], 'Measured'])

                #df = pandas.DataFrame(data, columns = ['Modes', 'Frequency', 'Type'])

                #seaborn.boxplot(x = 'Modes', y = 'Frequency', data = df)
                data = numpy.array(data)
                plt.boxplot(numpy.array(data).transpose())

                #ax1 = plt.gca()

                #for ll, meas, rr, tick in zip(ppl, self.data[s], ppr, range(len(self.data[s]))):
                #    ax1.text(tick + 1, ax1.get_ylim()[1] * 0.90, '{0:10.{3}f} {1:10.{3}f} {2:10.{3}f}'.format(ll, meas, rr, precision),
                #             horizontalalignment='center', rotation=45, size='x-small')
                plt.xlabel('Mode')
                plt.ylabel('Computed - Measured')
            else:
                print "For dataset {0}".format(s)
                print "{0:8s} {1:10s} {2:10s} {3:10s}".format("Outside", "2.5th %", "measured", "97.5th %")
                for ll, meas, rr in zip(ppl, self.data[s], ppr):
                    print "{0:8s} {1:10.{4}f} {2:10.{4}f} {3:10.{4}f}".format("*" if (meas < ll or meas > rr) else " ", ll, meas, rr, precision)

posterior_predictive(hmc)
gcf = plt.gcf()
gcf.set_size_inches(12, 8)
#%%
reload(rus)

hmc.set_timestepping(epsilon = epsilon * 2, L = 50)

hmc.sample(debug = False)

##%%
# Working on this -- not ready yet
reload(rus)

print hmc.saves()

#%%
reload(rus)

c11mc12, c11p2c12, c44 = sympy.symbols('c11mc12 c11p2c12 c44')

V = sympy.Matrix([[-1, -1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 1, 0, 0]])

Vinv = sympy.Matrix([[-1, -1, 2, 0, 0, 0],
                     [-1, 2, -1, 0, 0, 0],
                     [1, 1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0, 3],
                     [0, 0, 0, 0, 3, 0],
                     [0, 0, 0, 3, 0, 0]]) / 3.0

C = sympy.Matrix([[c11mc12, 0, 0, 0, 0, 0],
                  [0, c11mc12, 0, 0, 0, 0],
                  [0, 0, c11p2c12, 0, 0, 0],
                  [0, 0, 0, c44, 0, 0],
                  [0, 0, 0, 0, c44, 0],
                  [0, 0, 0, 0, 0, c44]])

C = V * C * Vinv

hmc = rus.HMC(N, density, X, Y, Z, data, C, { c11mc12 : 1.0, c11p2c12 : 4.0, c44 : 1.0, 'std' : 5.0 }, constrained_positive = [c44, 'std'])

hmc.set_labels({ c11mc12 : 'c11 - c12', c11p2c12 : 'c11 + 2 * c12', c44 : 'c44', 'std' : 'std' })
hmc.set_timestepping(epsilon = epsilon, L = 50)

hmc.sample(debug = True)
#%%
reload(rus)

hmc.set_timestepping(epsilon = epsilon * 20, L = 50)

hmc.sample(debug = False)
#%%
import matplotlib.pyplot as plt
import seaborn

for name, data1 in zip(*hmc.format_samples()):
    plt.plot(data1)
    plt.title('{0}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 24)
    plt.tick_params(axis='y', which='major', labelsize=16)
    plt.show()
#%%
for name, data1 in zip(*hmc.format_samples()):
    data1 = data1[-10000:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='x', which='major', labelsize=16)
    plt.show()