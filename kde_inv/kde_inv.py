__doc__ = "a module that houses basic functionality for kde-inversion playground stuff"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np
from scipy.optimize import minimize

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

def logleave1outLikelihood(bandwidth, samples, num_test, num_trials=10):
    """
    determine the likelihood of observing samples with Gaussian KDE using bandwidth

    return -logLike because we want to minimize this
    """
    loglike = 0.

    truth = np.ones_like(samples, dtype=bool)
    N = len(samples)
    for _ in xrange(num_trials):
        truth[:] = True
        inds = np.random.randint(0, N, size=num_test)
        truth[inds] = False
        loglike -= np.sum(logkde_pdf(samples[inds], samples[truth], bandwidth))/(N-np.sum(truth))

    loglike /= num_trials

    print 'bandwidth : %.6e'%bandwidth
    print '    -logLike : %.6e'%loglike

    return loglike

def grad_logleave1outLikelihood(bandwidth, samples, num_test, num_trials=10):
    """
    determine the gradient of the likelihood

    return -grad(logLike) becaues it needs to be consistent with logleave1outLikelihood
    """
    grad_loglike = 0.

    truth = np.ones_like(samples, dtype=bool)
    N = len(samples)
    for _ in xrange(num_trials):
        truth[:] = True
        inds = np.random.randint(0, N, size=num_test)
        truth[inds] = False
        grad_loglike -= np.sum(grad_logkde_pdf(samples[inds], samples[truth], bandwidth))/(N-np.sum(truth))

    grad_loglike /= num_trials

    print 'bandwidth : %.6e'%bandwidth
    print '    -grad_loglike : %.6e'%grad_loglike

    return grad_loglike

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
