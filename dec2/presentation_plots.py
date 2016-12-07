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

with open('dec2/andrew_cubic.pkl') as f:
    hmc = pickle.load(f)
    alabels, avalues = hmc.format_samples()

#%%
import scipy.stats

import matplotlib.pyplot as plt
import seaborn

for name1, name2, data1, data2 in zip(llabels, hlabels, lvalues, hvalues):
    plt.plot(data1)
    plt.title('{0}'.format(name1), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxlow/{0}t.png'.format(name1), dpi = 144)
    plt.show()

for name1, name2, data1, data2 in zip(llabels, hlabels, lvalues, hvalues):
    plt.plot(data2)
    plt.title('{0}'.format(name2), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxhigh/{0}t.png'.format(name2), dpi = 144)
    plt.show()

for name1, name2, data1, data2 in zip(llabels, hlabels, lvalues, hvalues):
    data1 = data1[-14000:]
    data2 = data2[-1000:]
    seaborn.distplot(data1, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name1, numpy.mean(data1), numpy.std(data1)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxlow/{0}d.png'.format(name1), dpi = 144)
    plt.show()

for name1, name2, data1, data2 in zip(llabels, hlabels, lvalues, hvalues):
    data1 = data1[-14000:]
    data2 = data2[-1000:]
    seaborn.distplot(data2, kde = False, fit = scipy.stats.norm)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name2, numpy.mean(data2), numpy.std(data2)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxhigh/{0}d.png'.format(name2), dpi = 144)
    plt.show()

#%%

for name, data in zip(alabels, avalues):
    plt.plot(data)
    plt.title('{0}'.format(name), fontsize = 36, color = "grey")
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/andrewcubic/{0}t.png'.format(name), dpi = 144)
    plt.show()

for name, data in zip(alabels, avalues):
    data = data[-14000:]
    seaborn.distplot(data, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name, numpy.mean(data), numpy.std(data)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/andrewcubic/{0}d.png'.format(name), dpi = 144)
    plt.show()
#%%

ranges = [[2.0, 7.5],
          [0.0, 2.0],
          [1.1, 1.4],
          [2.4, 3.1]]

for r, name1, name2, data1, data2 in zip(ranges, llabels[0:4], hlabels[0:4], lvalues[0:4], hvalues[0:4]):
    #data1 = data1[-14000:]
    data2 = data1[-14000:]#data2[-1000:]
    seaborn.distplot(data2, kde = False, fit = scipy.stats.norm, color = "grey")
    plt.xlim(r)
    plt.title('{0}, $\mu$ = {1:0.3f}, $\sigma$ = {2:0.3f}'.format(name2, numpy.mean(data2), numpy.std(data2)), fontsize = 36)
    plt.tick_params(axis='y', which='major', labelsize=24)
    plt.tick_params(axis='x', which='major', labelsize=24)
    fig = plt.gcf()
    fig.set_size_inches((10, 6.6667))
    plt.savefig('dec2/cmsxhighlow/{0}dl.png'.format(name2), dpi = 144)
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