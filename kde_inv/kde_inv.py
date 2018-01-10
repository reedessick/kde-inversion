__doc__ = "a module that houses basic functionality for kde-inversion playground stuff"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

def drawA(size=1):
    """
    draw samples from p(A)
    """
    raise NotImplementedError

def mapA2B(A):
    """
    the mapping from A -> B
    """
    raise NotImplementedError

def optimizeBandwidth(samples):
    """
    find the optimal bandwidth to describe samples with a Gaussian KDE via leave-one-out cross validation
    """
    raise NotImplementedError

def leave1outLikelihood(samples, bandwidth):
    """
    determine the likelihood of observing samples with Gaussian KDE using bandwidth
    """
    raise NotImplementedError

def kldiv(samples, foo):
    """
    compute the KL Divergence between a selection of samples drawn from a distribution and an estimate of that distribution
    will be used to measure how well we can recover a distribution
    """
    raise NotImplementedError
