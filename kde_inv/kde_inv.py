__doc__ = "a module that houses basic functionality for kde-inversion playground stuff"
__author__ = "Reed Essick (reed.essick@ligo.org)"

#-------------------------------------------------

def drawA(size=1):
    """
    draw samples from p(A)

    take p(A) = p(A[0])*p(A[1]) = 1
    """
    raise NotImplementedError

def mapA2B(A):
    """
    the mapping from A -> B

    take B = np.sin(2*np.pi*A[0])**2 * np.cos(4*np.pi*A[1])**2
    for A[0], A[1] \in [0, 1]
    """
    raise NotImplementedError

def optimizeBandwidth(samples):
    """
    find the optimal bandwidth to describe samples with a Gaussian KDE via leave-one-out cross validation

    bisection search with starting boundary points as the closest distance between 2 samples and the furthest distance between 2 samples?
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
