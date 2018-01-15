__doc__ = "a module that houses basic functionality for kde-inversion playground stuff"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import os

import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

#-------------------------------------------------

def drawA(size=1):
    """
    draw samples from p(A)

    take p(A) = p(A[0])*p(A[1]) = 1
    for A[0], A[1] \in [0, 1]
    """
    return np.random.random(size=(2,size))

def mapA2B(A):
    """
    the mapping from A -> B = A[0] + np.cos(4*np.pi*A[1])**2
    for A[0], A[1] \in [0, 1]
    """
    return A[0] + np.cos(4*np.pi*A[1])**2

#-------------------------------------------------

def likelihood_plot(samples, bandwidths, num_test, num_trials, num_plot=101, verbose=False):
    """
    returns a figure showing the estimates for likelihood, grad_likelihood, and KDE representations of the histograms
    """
    ### set up figures and axes
    fig = plt.figure()
    ax_logL = plt.subplot(2, 2, 1)
    ax_gradlogL = plt.subplot(2, 2, 3)
    ax_hist = plt.subplot(1, 2, 2)

    ax_logL.set_xlabel('bandwidth')
    ax_logL.set_ylabel('logL')

    ax_gradlogL.set_xlabel('bandwidth')
    ax_gradlogL.set_ylabel('d(logL)/d(logbandwidth)')

    ax_hist.set_xlabel('B')
    ax_hist.set_ylabel('p(B)')
    ax_hist.set_xlim(xmin=0, xmax=2)

    for ax in [ax_logL, ax_gradlogL, ax_hist]:
        ax.grid(True, which='both')

    xlim = min(bandwidths), max(bandwidths)
    for ax in [ax_logL, ax_gradlogL]:
        ax.set_xlim(xlim)

    ### generate histogram over samples
    N = len(samples)
    ax_hist.hist(samples, bins=int(N**0.5), histtype='step', normed=True, label='samples', color='k', linewidth=2)

    ### iterate over bandwidths, adding to plots as we go
    b = np.linspace(0, 2, num_plot)
    for bandwidth in bandwidths:
        if verbose:
            print('bandwidth=%.3e'%bandwidth)
            print('    computing logL')
        logL, sigma_logL = logleave1outLikelihood(bandwidth, samples, num_test, num_trials, return_std=True)

        if verbose:
            print('    computing grad_logL')
        gradlogL, sigma_gradlogL = grad_logleave1outLikelihood(bandwidth, samples, num_test, num_trials, return_std=True)

        if verbose:
            print('    plotting')
        color = ax_logL.semilogx([bandwidth]*2, [logL-sigma_logL, logL+sigma_logL], alpha=0.5)[0].get_color()
        ax_logL.semilogx(bandwidth, logL, marker='o', color=color)

        ax_gradlogL.semilogx([bandwidth]*2, [bandwidth*(gradlogL-sigma_gradlogL), bandwidth*(gradlogL+sigma_gradlogL)], alpha=0.5, color=color)
        ax_gradlogL.semilogx(bandwidth, gradlogL*bandwidth, marker='o', color=color)

        ax_hist.plot(b, np.exp(logkde_pdf(b, samples, bandwidth)), label='bandwidth=%.3e'%bandwidth, alpha=0.5, color=color)

        fig.savefig('tmp.png')

#    ax_hist.legend(loc='best')
    os.remove('tmp.png')

    return fig

def optimizeBandwidth(samples, num_test=100, num_trials=10, rtol=1e-3, method='Newton-CG'):
    """
    find the optimal bandwidth to describe samples with a Gaussian KDE via leave-one-out cross validation

    bisection search with starting boundary points as the closest distance between 2 samples and the furthest distance between 2 samples?
    """
    ### find starting points for bandwidth search
    order = samples.argsort() ### smallest to largest
    bandwidth = np.mean(samples[order][1:]-samples[order][:-1])

    ### delegate for minimization
    res = minimize(
        logleave1outLikelihood,
        bandwidth,
        args=(samples, num_test, num_trials),
        tol=rtol,
        jac=grad_logleave1outLikelihood,
        method=method,
    )
    assert res.success, 'minimization did not complete succesfully!'
    return res.x ### return the optimal bandwidth

def logleave1outLikelihood(bandwidth, samples, num_test, num_trials=10, return_std=False):
    """
    determine the likelihood of observing samples with Gaussian KDE using bandwidth

    return -logLike because we want to minimize this
    """
    truth = np.ones_like(samples, dtype=bool)
    N = len(samples)
    loglike = []
    for _ in xrange(num_trials):
        truth[:] = True
        inds = np.random.randint(0, N, size=num_test)
        truth[inds] = False
        loglike.append( -np.sum(logkde_pdf(samples[inds], samples[truth], bandwidth))/(N-np.sum(truth)) )

    if return_std:
        return np.mean(loglike), np.std(loglike)
    else:
        return np.mean(loglike)

'''
### DEPRECATED: this is actually slower than logleave1outLikelihood for reasonable sample sizes
def fast_logleave1outLikelihood(bandwidth, samples, num_test, num_trials=10):
    N = len(samples)
    sample_inds = np.arange(N)
    ones_N = np.ones_like(samples, dtype=float)
    ones_T = np.ones(num_trials, dtype=float)

    ### compute all kernals between samples
    kernals = (2*np.pi*bandwidth)**-0.5 * np.exp(-0.5*(np.outer(samples, ones_N) - np.outer(ones_N, samples))**2/bandwidth)

    ### randomly partition these into test and train sets
    train = np.ones((num_trials,N), dtype=bool) ### shape : (num_trials, N)
    inds = np.random.randint(0, N, size=num_trials*num_test).reshape((num_trials, num_test))
    for k in xrange(num_trials):
        train[k][inds[k]] = False ### select the subset for testing
    num_train = np.outer(np.sum(train, axis=1), ones_N) ### shape : (num_trials, N)

    test = np.logical_not(train)
    num_test = np.sum(test, axis=1) ### shape : (num_trials,)

    ### set up indicator function -> set only those elements that are included to non-zero values
    training = np.zeros((num_trials, N, N), dtype=float) ### if not selected, this element should just be zero
    for k in xrange(num_trials):
        for i in sample_inds[test[k]]:
            training[k][i][train[k]] = kernals[i][train[k]]

    ### set up the summation over train samples
    summed_over_train = np.log(np.sum(training, axis=2)) - np.log(num_train)

    ### sum over only stuff in test set
    for k in xrange(num_trials):
        summed_over_train[k][train[k]] = 0 ### zero out the stuff that isn't in the test set
    summed_over_test  = np.sum(summed_over_train, axis=1)/num_test

    ### average over trials
    summed_over_trials = np.sum(summed_over_test, axis=0)/num_trials

    return -summed_over_trials ### return the negative because we want to minimize this...
'''

def grad_logleave1outLikelihood(bandwidth, samples, num_test, num_trials=10, return_std=False):
    """
    determine the gradient of the likelihood

    return -grad(logLike) becaues it needs to be consistent with logleave1outLikelihood
    """
    truth = np.ones_like(samples, dtype=bool)
    N = len(samples)
    grad_loglike = []
    for _ in xrange(num_trials):
        truth[:] = True
        inds = np.random.randint(0, N, size=num_test)
        truth[inds] = False
        grad_loglike.append( -np.sum(grad_logkde_pdf(samples[inds], samples[truth], bandwidth))/(N-np.sum(truth)) )

    if return_std:
        return np.mean(grad_loglike), np.std(grad_loglike)
    else:
        return np.mean(grad_loglike)

def fast_grad_logleave1outLikelihood(bandwidth, samples, num_test, num_trials=10):
    """
    a version of grad_logleave2outLikelihood that uses numpy arrays to try to speed up the calculation
    """
    raise NotImplementedError

def logkde_pdf(B, samples, bandwidth):
    """
    compute pdf at sample point B from KDE estimate of samples with bandwidth

    NOTE: returns an array with the same shape as B
    """
    N = 1.*len(samples)
    samples = np.outer(np.ones_like(B), samples)
    B = np.outer(B, np.ones(N))
    return np.log((2*np.pi*bandwidth)**-0.5 * np.sum(np.exp(-0.5*(B-samples)**2/bandwidth), axis=1)/N)

def grad_logkde_pdf(B, samples, bandwidth):
    """
    NOTE: returns an array with the same shape as B
    """
    N = 1.*len(samples)
    samples = np.outer(np.ones_like(B), samples)
    B = np.outer(B, np.ones(N))
    z = (B-samples)**2/bandwidth
    e = np.exp(-0.5*z)
    return np.sum(e*(z - 1), axis=1)/(np.sum(e, axis=1)*2*bandwidth)

def kldiv(samples, foo):
    """
    compute the KL Divergence between a selection of samples drawn from a distribution and an estimate of that distribution
    will be used to measure how well we can recover a distribution
    """
    raise NotImplementedError
