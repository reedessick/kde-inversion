# kde-inversion

A playground repo for testing ideas about "inverting" KDE density estimates.

Assume we have some complicated mapping `f: A -> B`, where the dimensions of `A` and `B` do not have to agree, `f` does not have to one-to-one (not invertible), and can generally have a very complicated shape. Now, assume that we can define *natural* prior distributions over `A`. To calculate the associated distribution over `B`, we have

> p(B) = \int dA p(A) p(B|A) = \int dA \delta(B - f(A))

In general, this integral can be difficult, if not impossible, to do analytically. A natural approach is to sample the distribution `p(A)`, compute the associated value of `B`, and then use the samples to respresent draws from `p(B)`.

An interesting inverse problem involves taking a likelihood defined as a set of samples over `B` and using it to compute a posterior distribution over `A`. If the form of the likelihood is known, one can always simply sample from the distribution

> L(data|B=f(A))p(A)

However, this may be computationally expensive, particularly if one already has samples that are drawn from `L(data|B)`. This module explores the possiblity of approximating the inference problem for `A` using samples from `L(data|B)` and an approximate, invertable mapping `K(B, B(A))` based on Kernal Density Estimation.

The basic idea is that we can approximate `p(B)` via KDE with samples (`i`) drawn from `p(A)` via

> p(B) \approx \sum_i K(B, f(A_i))

where `K(B, f(A_i))` is the kernal function, typically taken to be a Gaussian with some fixed bandwidth (standard deviation). As the number of `i` samples diverges, we expect the optimal bandwidth to shrink but eventually converge to something like the natural distance scale within `p(B)`. If that convergence occurs, which is not guaranteed, then it may be natural to associate

> p(B|A) <==> K(B, f(A))

The "fuzz" introduced through `K(B, f(A))` should help get around the issue of discrete samples and the actual mapping `p(B|A)=\delta(B-f(A))`.

This module will explore whether any of these ideas hold water.

### optimal bandwidths

We can define the optimal bandwidth via *leave-one-out* cross validation. We maximize the likelihood of drawing each sample given an estimate of `p(B)` generated using the rest of the samples with respect to the bandwidth. If the bandwidth is too small, the KDE will be over-trained and result in a small likelihood. If the bandwidth is too large, the KDE will be smeared out more than it needs to be and the likelihood can be increased by tightening up the distribution (shortening the tails).
