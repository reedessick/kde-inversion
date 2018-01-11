__doc__ = "a module that houses basic functionality for kde-inversion playground stuff"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

import numpy as np

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
    the mapping from A -> B

    take B = np.sin(2*np.pi*A[0])**2 * np.cos(4*np.pi*A[1])**2
    for A[0], A[1] \in [0, 1]
    """
    return np.abs(np.sin(2*np.pi*A[0]) * np.cos(4*np.pi*A[1]))

def optimizeBandwidth(samples, rtol=1e-6):
    """
    find the optimal bandwidth to describe samples with a Gaussian KDE via leave-one-out cross validation

    bisection search with starting boundary points as the closest distance between 2 samples and the furthest distance between 2 samples?
    """
    max_bandwidth = np.max(samples) - np.min(samples)
    order = samples.argsort()
    min_samples = np.min(samples[order][1:]-samples[order][:-1])

    raise NotImplementedError, 'implement a bisection search with truncation condition on relative improvement given by rtol'

def leave1outLikelihood(samples, bandwidth):
    """
    determine the likelihood of observing samples with Gaussian KDE using bandwidth
    """
    pdf = 1.0
    truth = np.ones_like(samples, dtype=bool)
    for i in xrange(len(samples)):
        truth[i-1] = True
        truth[i] = False
        pdf *= kde_pdf(samples[i], samples[truth], bandwidth)
    return pdf

def kde_pdf(B, samples, bandwidth):
    """
    compute pdf at sample point B from KDE estimate of samples with bandwidth
    """
    return (2*np.pi*bandwidth)**-0.5 * np.sum(np.exp(-0.5*(B-samples)**2/bandwidth))/len(bandwidth)

def kldiv(samples, foo):
    """
    compute the KL Divergence between a selection of samples drawn from a distribution and an estimate of that distribution
    will be used to measure how well we can recover a distribution
    """
    raise NotImplementedError
