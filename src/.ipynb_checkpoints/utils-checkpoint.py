import torch
from gpytorch.distributions import MultivariateNormal
from torch.distributions.kl import _batch_mahalanobis, _batch_trace_XXT


def expect_kl(p:MultivariateNormal, q:MultivariateNormal, expect_inverse):
    """Adapt from Pytorch github in the following link:
    https://github.com/pytorch/pytorch/blob/d83509e603d6d932a818d4e0ca027129aa9c5857/torch/distributions/kl.py#L384

    NOTE: the entropy terms for Horseshoe variables are removed because they do not affect optimization for GP
    """

    if p.event_shape != q.event_shape:
        raise ValueError("KL-divergence between two Multivariate Normals with\
                              different event shapes cannot be computed")

    a = q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    b = p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
    half_term1 = (q._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1) -
                  p._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1))
    combined_batch_shape = torch._C._infer_size(q._unbroadcasted_scale_tril.shape[:-2],
                                                p._unbroadcasted_scale_tril.shape[:-2])
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_scale_tril = p._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    term2 = _batch_trace_XXT(torch.triangular_solve(p_scale_tril, q_scale_tril, upper=False)[0])
    term3 = _batch_mahalanobis(q._unbroadcasted_scale_tril, (q.loc - p.loc))

    # print(a)
    # print(b)
    # if b != b:
    #     print("debug")

    return half_term1 + 0.5 * expect_inverse *(term2 + term3) - 0.5 * n

def trace_stats(p, q):
    if p.event_shape != q.event_shape:
        raise ValueError("KL-divergence between two Multivariate Normals with\
                              different event shapes cannot be computed")

    combined_batch_shape = torch._C._infer_size(q._unbroadcasted_scale_tril.shape[:-2],
                                                p._unbroadcasted_scale_tril.shape[:-2])
    n = p.event_shape[0]
    q_scale_tril = q._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    p_scale_tril = p._unbroadcasted_scale_tril.expand(combined_batch_shape + (n, n))
    return _batch_trace_XXT(torch.triangular_solve(p_scale_tril, q_scale_tril, upper=False)[0])

