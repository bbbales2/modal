#%%

import os
import pystan
import numpy
import pickle
import matplotlib.pyplot as plt

os.chdir('/home/bbales2/modal/')

with open('dec2/friday_demo_lowres.pkl') as f:
    llabels, lvalues = pickle.load(f)

with open('dec2/friday_demo_hires.pkl') as f:
    hlabels, hvalues = pickle.load(f)

with open('dec2/low16res') as f:
    hmcl = pickle.load(f)
    llabels, lvalues = hmcl.format_samples()

with open('dec2/hires') as f:
    hmch = pickle.load(f)
    hlabels, hvalues = hmch.format_samples()

with open('dec2/andrew_cubic.pkl') as f:
    hmcc = pickle.load(f)
    aclabels, acvalues = hmcc.format_samples()

hmcc.dp = [hmcc.dp]
hmcc.pv = [hmcc.pv]
hmcc.density = [hmcc.density]

with open('dec2/andrew_hexagonal.pkl') as f:
    hmcf = pickle.load(f)
    ahlabels, ahvalues = hmcf.format_samples()

hmcf.dp = [hmcf.dp]
hmcf.pv = [hmcf.pv]
hmcf.density = [hmcf.density]

#%%
import scipy.stats

import matplotlib.pyplot as plt
import seaborn

for name1, data1 in zip(llabels, lvalues):
    plt.plot(data1)
    plt.title('{0}'.format(name1), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxlow/{0}t.png'.format(name1), dpi = 144)
    plt.show()

for name1, data1 in zip(llabels, lvalues):
    data1 = data1[-1000:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name1, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxlow/{0}d.png'.format(name1), dpi = 144)
    plt.show()
#%%
for name2, data2 in zip(hlabels, hvalues):
    plt.plot(data2)
    plt.title('{0}'.format(name2), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxhigh/{0}t.png'.format(name2), dpi = 144)
    plt.show()

for name2, data2 in zip(hlabels, hvalues):
    data2 = data2[-1000:]
    seaborn.distplot(data2, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name2, numpy.mean(data2), numpy.std(data2)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxhigh/{0}d.png'.format(name2), dpi = 144)
    plt.show()

#%%

for name, data in zip(aclabels, acvalues):
    plt.plot(data)
    plt.title('{0}'.format(name), fontsize = 36, color = "grey")
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/andrewcubic/{0}t.png'.format(name), dpi = 144)
    plt.show()

for name, data in zip(aclabels, acvalues):
    data = data[-2000:]
    seaborn.distplot(data, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data), numpy.std(data)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/andrewcubic/{0}d.png'.format(name), dpi = 144)
    plt.show()
#%%
hmcc.posterior_predictive(plot = True, lastN = 200)
plt.title('Posterior predictive', fontsize = 72)
plt.xlabel('Mode', fontsize = 48)
plt.ylabel('Computed - Measured (khz)', fontsize = 48)
plt.tick_params(axis='y', which='major', labelsize=48)
plt.tick_params(axis='x', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches((24, 16))
plt.savefig('dec2/andrewcubic/posteriorpredictive.png', dpi = 144)
plt.show()
#%%
hmcl.posterior_predictive(plot = True, lastN = 200)
plt.title('Posterior predictive', fontsize = 72)
plt.xlabel('Mode', fontsize = 48)
plt.ylabel('Computed - Measured (khz)', fontsize = 48)
plt.tick_params(axis='y', which='major', labelsize=48)
plt.tick_params(axis='x', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches((24, 16))
plt.savefig('dec2/cmsxlow/posteriorpredictive.png', dpi = 144)
plt.show()
#%%
hmch.posterior_predictive(plot = True, lastN = 200)
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

for name, data in zip(ahlabels, ahvalues):
    plt.plot(data)
    plt.title('{0}'.format(name), fontsize = 36, color = "grey")
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/andrewhexagonal/{0}t.png'.format(name), dpi = 144)
    plt.show()

for name, data in zip(ahlabels, ahvalues):
    data = data[-14000:]
    seaborn.distplot(data, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data), numpy.std(data)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/andrewhexagonal/{0}d.png'.format(name), dpi = 144)
    plt.show()
#%%
hmcf.posterior_predictive(plot = True, lastN = 200)
plt.title('Posterior predictive', fontsize = 72)
plt.xlabel('Mode', fontsize = 48)
plt.ylabel('Computed - Measured (khz)', fontsize = 48)
plt.tick_params(axis='y', which='major', labelsize=48)
plt.tick_params(axis='x', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches((24, 16))
plt.savefig('dec2/andrewhexagonal/posteriorpredictive.png', dpi = 144)
plt.show()
#%%

ranges = [[2.2, 3.1],
          [0.0, 0.5],
          [2.8, 2.95],
          [1.28, 1.35]]

for r, name1, name2, data1, data2 in zip(ranges, llabels[0:4], hlabels[0:4], lvalues[0:4], hvalues[0:4]):
    #data1 = data1[-14000:]
    data2 = data1[-1000:]#
    seaborn.distplot(data2, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.xlim(r)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name2, numpy.mean(data2), numpy.std(data2)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxhighlowCompare/{0}dl.png'.format(name2), dpi = 144)
    plt.show()

for r, name1, name2, data1, data2 in zip(ranges, llabels[0:4], hlabels[0:4], lvalues[0:4], hvalues[0:4]):
    #data1 = data1[-14000:]
    data2 = data2[-1000:]#
    seaborn.distplot(data2, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.xlim(r)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name2, numpy.mean(data2), numpy.std(data2)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxhighlowCompare/{0}dh.png'.format(name2), dpi = 144)
    plt.show()

#%%
qs = numpy.array([q for q in zip(*values[0:8])[-10000:]])

#%%
import sklearn.mixture

gmm = sklearn.mixture.GMM(3, tol = 1e-12, min_covar = 1e-12)

gmm.fit(numpy.array(lvalues)[0:1, -14000:].T)
print gmm.weights_
print gmm.means_
print numpy.sqrt(gmm.covars_)

idxs = numpy.argsort(gmm.means_.flatten())
covars = gmm.covars_[idxs]
weights = gmm.weights_[idxs]
means = gmm.means_[idxs]

    #plt.hist
xs = numpy.linspace(ranges[0][0], ranges[0][1], 500)
labels = []
seaborn.distplot(lvalues[0][-14000:], kde = False, norm_hist = True, color = 'grey')
for w, m, s in zip(weights, means, numpy.sqrt(covars)):
    plt.plot(xs, w * scipy.stats.norm.pdf(xs, m[0], s[0]), linewidth = 4)
    labels.append("$\mu = {0:0.3f}$".format(m[0]))

plt.xlim((2.0, 7.5))
plt.title('{0}'.format('c11'), fontsize = 36)
plt.tick_params(axis='y', which='major', labelsize=24)
plt.tick_params(axis='x', which='major', labelsize=24)
plt.legend(labels, fontsize=24)
fig = plt.gcf()
fig.set_size_inches((10, 6.6667))
plt.savefig('dec2/cmsxhighlow/fitted.png'.format(name2), dpi = 144)
plt.show()
#%%
elasticities = qs[:, [0, 2, 3]]
plt.hist(elasticities, bins = 200, histtype = 'stepfilled', alpha = 0.5, normed = True)

plt.legend([labels[3], labels[2], labels[0]])
#%%
angles = qs[:, 4:]
plt.hist(angles, bins = 200, histtype = 'stepfilled', alpha = 0.5, normed = True)

plt.legend(labels[7], labels, loc="center")

#ax = plt.hist(qs, normed = True, bins = 200, histtype = 'stepfilled', label = labels)

#%%

model_string = """
data {
    int<lower=1> K; // number of mixture components
    int<lower=1> N; // number of data points
    int<lower=1> D; // dimension of observations
    real y[N, D]; // observations
}

parameters {
    simplex[K] theta; // mixing proportions
    real mu[K, D]; // locations of mixture components
    real<lower=0> sigma[K, D]; // scales of mixture components
}

model {
    real ps[K]; // temp for log component densities

    for (d in 1:D) {
        for(k in 1:K) {
            sigma[k, d] ~ cauchy(0, 10.0);
            mu[k, d] ~ uniform(0.0, 4.0);
        }
    }

    for (n in 1:N) {
        for (d in 1:D) {
            for (k in 1:K) {
                ps[k] = log(theta[k]) + cauchy_lpdf(y[n, d] | mu[k, d], sigma[k, d]);
            }

            target += log_sum_exp(ps);
        }
    }
}

generated quantities {

}
"""

sm = pystan.StanModel(model_code = model_string)

#%%


fit = sm.sampling(data = {
    'K' : 3,
    'N' : len(qs),
    'D' : 1,
    'y' : qs[:, :1]
})

print fit
#%%
#%%
import polybasisqu

def posterior_predictive(qs, data, lastN = 200, precision = 5, plot = True, which_samples = None):
        lastN = min(lastN, len(qs))

        modes = len(data)

        posterior_predictive = numpy.zeros((modes, lastN, 1))

        if which_samples == None:
            which_samples = range(1)

        for i, q in enumerate(qs[-lastN:]):
            for s in which_samples:
                c11 = q[0]
                std = q[1]
                a = q[2]
                c44 = q[3]

                c12 = -(c44 * 2.0 / a - c11)

                C = numpy.array([[c11, c12, c12, 0, 0, 0],
                                  [c12, c11, c12, 0, 0, 0],
                                  [c12, c12, c11, 0, 0, 0],
                                  [0, 0, 0, c44, 0, 0],
                                  [0, 0, 0, 0, c44, 0],
                                  [0, 0, 0, 0, 0, c44]])

                w, x, y, z = q[4:]

                C, _, _, _, _, _ = polybasisqu.buildRot(C, w, x, y, z)

                K, M = polybasisqu.buildKM(C, dp, pv, density)

                eigs, evecs = scipy.linalg.eigh(K, M, eigvals = (6, 6 + modes - 1))

                posterior_predictive[:, i, s] = numpy.sqrt(eigs * 1e11) / (numpy.pi * 2000) + numpy.random.randn() * std
        #print l, r, posterior_predictive[0]

        print posterior_predictive
#
##%%
        for s in [0]:
            r = 0

            ppl = numpy.percentile(posterior_predictive[:, :, s], 2.5, axis = 1)
            ppr = numpy.percentile(posterior_predictive[:, :, s], 97.5, axis = 1)

            if plot:
                generated = []

                for l in range(len(data)):
                    tmp = []
                    for ln in range(lastN):
                        tmp.append(posterior_predictive[l, ln, s] - data[l])

                    generated.append(tmp)

                generated = numpy.array(generated)
                plt.boxplot(numpy.array(generated).transpose())

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

#%%

qs = zip(*lvalues)

N = 8

## Dimensions for TF-2
X = 0.011959#0.007753
Y = 0.013953#0.009057
Z = 0.019976#0.013199

density = 8700.0

data = numpy.array([71.25925,
75.75875,
86.478,
89.947375,
111.150125,
112.164125,
120.172125,
127.810375])

dp, pv, _, _, _, _, _, _ = polybasisqu.build(N, X, Y, Z)

posterior_predictive(qs, data, 200)
plt.title('Posterior predictive', fontsize = 72)
plt.xlabel('Mode', fontsize = 48)
plt.ylabel('Computed - Measured (khz)', fontsize = 48)
plt.tick_params(axis='y', which='major', labelsize=48)
plt.tick_params(axis='x', which='major', labelsize=16)
fig = plt.gcf()
fig.set_size_inches((24, 16))
plt.savefig('dec2/cmsxlow/posteriorpredictive.png', dpi = 144)
plt.show()

#%%

qs = zip(*[hvalues[0], hvalues[1], hvalues[3], hvalues[2], hvalues[4], hvalues[5], hvalues[6], hvalues[7]])

N = 12

## Dimensions for TF-2
X = 0.011959#0.007753
Y = 0.013953#0.009057
Z = 0.019976#0.013199

density = 8700.0

data = numpy.array([71.25925,
75.75875,
86.478,
89.947375,
111.150125,
112.164125,
120.172125,
127.810375,
128.6755,
130.739875,
141.70025,
144.50375,
149.40075,
154.35075,
156.782125,
157.554625,
161.0875,
165.10325,
169.7615,
173.44925,
174.11675,
174.90625,
181.11975,
182.4585,
183.98625,
192.68125,
193.43575,
198.793625,
201.901625,
205.01475,
206.619,
208.513875,
208.83525,
212.22525,
212.464125,
221.169625,
225.01225,
227.74775,
228.31175,
231.4265,
235.792875,
235.992375,
236.73675,
238.157625,
246.431125,
246.797125,
248.3185,
251.69425,
252.97225,
253.9795,
256.869875,
258.23825,
259.39025])

dp, pv, _, _, _, _, _, _ = polybasisqu.build(N, X, Y, Z)

posterior_predictive(qs, data, 200)
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
#hmc.set_timestepping(epsilon = epsilon, L = 50)
hmcl.set_timestepping(epsilon = 0.0001 * 25, L = 100)

hmcl.sample(100, debug = False)