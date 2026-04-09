# Torch multimodal gaussian support is quite cringe. This file implements some basic distribution functions (pdf, logpdf, and sample) with standardized syntax,
# supporting both scalar and matrix sigmas
#TODO: rigorous testing of the numpy preservation from torch_fn

from typing import Literal, Optional
import numpy as np
import torch
from torch import Tensor
from diffenergy.torch_decorator import torch_fn

@torch_fn #coerce inputs to tensors
def mixture_sample(num_samples:int, weights:Tensor, means:Tensor, variances:Optional[Tensor]=None, stds:Optional[Tensor]=None)->Tensor:
    """
    Sample num_samples points from a D-dimensional N-modal gaussian mixture model parameterized by the weights for each model (N-length tensor),
    a sqeuence of means (NxD tensor), and a sequence of either (co)variances or standard deviations. The matrix "standard deviation" is the lower triangular matrix which squares to the covariance matrix (that is, S @ S.T = V) whose diagonal entries are positive.
    If the covariances are zero but each dimension has unique variances, the "standard deviation" matrix has a diagonal of the standard deviations for each dimension. 
    See torch.distributions.MultivariateNormal's scale_tril parameter for more details.

    For scalar-valued variances/stds, providing variances is preferred, whereas lower triangular standard deviation matrices are preferred over covariance matrices
    to avoid a costly (and cpu-only!) matrix decomposition for inversion.
    
    :param num_samples: Number of samples to draw from the mixture model
    :type num_samples: int
    :param weights: Scalar weight of each of the N gaussians, shape (N,)
    :type weights: Tensor
    :param means: Means of the N D-dimensional gaussians, shape (N,D)
    :type means: Tensor
    :param variances: (Co)variances of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type variances: Optional[Tensor]
    :param stds: Standard deviations of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type stds: Optional[Tensor]
    :return: Tensor of D-dimensional samples. Shape (num_samples,D)
    :rtype: Tensor
    """
    # weights = torch.as_tensor(weights)
    # means = torch.as_tensor(means)
    # variances = torch.as_tensor(variances) if variances is not None else variances
    # stds = torch.as_tensor(stds) if stds is not None else stds

    if means.ndim != 2:
        raise ValueError(f"means must be a 2D tensor with shape (N,D), got {means.shape}")
    if not (variances is None) ^ (stds is None): #exclusive or, exactly one of variances and stds is provided
        raise ValueError("Exactly one of variances and stds must be specified, the other must be None")
    if variances is not None:
        var_shape = variances.shape
    elif stds is not None:
        var_shape = stds.shape
    else:
        raise Exception() #should be unreachable
    
    
    N = var_shape[0]
    if not means.shape[0] == N:
        raise ValueError(f"Variance/deviation shape {var_shape} incompatible with means shape {means.shape}, first dimension (number of distributions) must match!")
    
    D = means.shape[-1]

    is_scalar = True
    if len(var_shape) == 3: #matrix
        is_scalar = False
        if not var_shape[-1] == D:
            raise ValueError(f"Means shape {means.shape} incompatible with variance/std shape {var_shape}, last dimension (dimension of distributions) must match!")
    elif len(var_shape) != 1:
        raise ValueError(f"Invalid variance/std shape {var_shape}; variance/std must be a sequence of scalars (N-tensor) or a sequence of DxD matrices (NxDxD-tensor)")

    # Since MultivariateNormal just transforms the result of a standard normal by multiplying by the standard deviation (lower triangular) matrix, 
    # providing scale_tril individually for each sample incurs basically no overhead besides memory. Thus, we stard by square rooting / decomposing the variances
    # for each distribution (if necessary), turning the scalar std into a matrix (if necessary), sampling from the categorical model to pick from the mixture,
    # and calling MultivariateNormal.sample() for each distribution in turn based on the categorical samples. Note that we still don't use the built-in 
    # mixture distribution since it samples from every distribution for every point *before* selecting based on the results of the categorical, which is wasteful
    
    # Square root / cholesky decompose variances into stds
    if stds is None: 
        assert variances is not None
        if is_scalar:
            stds = torch.sqrt(variances)
        else:
            stds,_ = torch.linalg.cholesky_ex(variances)
            assert stds is not None

    # Turn scalar stds into diagonal matrices
    # Note that we don't do this for computing the pdf since the generalized inversion method doesn't check for diagonality, but for sampling, 
    # it's just forward multiplication, so it's fast to just always use MultivariateNormal. 
    if len(var_shape) == 1:
        stds = torch.diag_embed(stds[:,None].expand(-1,D)) 

    choices = torch.distributions.OneHotCategorical(weights).sample((num_samples,)).T.to(dtype=bool)  #transpose of one-hot = masks!
    assert choices.shape[0] == N

    res = torch.empty((num_samples,D),dtype=means.dtype,device=means.device)
    for i in range(N):
        res[choices[i],:] = torch.distributions.MultivariateNormal(means[i],scale_tril=stds[i]).sample((int(choices[i].sum()),))
    
    return res

@torch_fn #coerce inputs to tensors
def mixture_logpdf(x:Tensor, weights:Tensor, means:Tensor|Literal[0], variances:Optional[Tensor]=None, stds:Optional[Tensor]=None): #x: BxD, weights: N, means: NxD, variances: N or NxDxD, stds: N or NxDxD, result: B
    """
    Computes the pdf at B D-dimensional points x (BxD tensor) of a N-modal D-dimensional gaussian mixture model parameterized by the weights for each model (N-length tensor),
    a sqeuence of means (NxD tensor), and a sequence of either (co)variances or standard deviations. The matrix "standard deviation" is the lower triangular matrix which squares to the covariance matrix (that is, S @ S.T = V) whose diagonal entries are positive.
    If the covariances are zero but each dimension has unique variances, the "standard deviation" matrix has a diagonal of the standard deviations for each dimension. 
    See torch.distributions.MultivariateNormal's scale_tril parameter for more details.

    For scalar-valued variances/stds, providing variances is preferred, whereas lower triangular standard deviation matrices are preferred over covariance matrices
    to avoid a costly (and cpu-only!) matrix decomposition for inversion.
    
    :param x: Points for which to calculate the log pdf. Shape (B,D) or (B,N,D)
    :type x: Tensor
    :param weights: Scalar weight of each of the N gaussians, shape (N,)
    :type weights: Tensor
    :param means: Means of the N D-dimensional gaussians, shape (N,D)
    :type means: Tensor|Literal[0]
    :param variances: (Co)variances of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type variances: Optional[Tensor]
    :param stds: Standard deviations of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type stds: Optional[Tensor]
    :return: pdf for each point in x. Shape (B,)
    :rtype: Tensor
    """
    pdf = mixture_pdf(x,weights,means,variances=variances,stds=stds)
    return np.log(pdf)

@torch_fn #coerce inputs to tensors
def mixture_pdf(x:Tensor, weights:Tensor, means:Tensor|Literal[0], variances:Optional[Tensor]=None, stds:Optional[Tensor]=None): #x: BxD, weights: N, means: NxD, variances: N or NxDxD, stds: N or NxDxD, result: B
    """
    Computes the pdf at B D-dimensional points x (BxD tensor) of a N-modal D-dimensional gaussian mixture model parameterized by the weights for each model (N-length tensor),
    a sqeuence of means (NxD tensor), and a sequence of either (co)variances or standard deviations. The matrix "standard deviation" is the lower triangular matrix which squares to the covariance matrix (that is, S @ S.T = V) whose diagonal entries are positive.
    If the covariances are zero but each dimension has unique variances, the "standard deviation" matrix has a diagonal of the standard deviations for each dimension. 
    See torch.distributions.MultivariateNormal's scale_tril parameter for more details.

    For scalar-valued variances/stds, providing variances is preferred, whereas lower triangular standard deviation matrices are preferred over covariance matrices
    to avoid a costly (and cpu-only!) matrix decomposition for inversion.
    
    :param x: Points for which to calculate the log pdf. Shape (B,D) or (B,N,D)
    :type x: Tensor
    :param weights: Scalar weight of each of the N gaussians, shape (N,)
    :type weights: Tensor
    :param means: Means of the N D-dimensional gaussians, shape (N,D)
    :type means: Tensor|Literal[0]
    :param variances: (Co)variances of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type variances: Optional[Tensor]
    :param stds: Standard deviations of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type stds: Optional[Tensor]
    :return: pdf for each point in x. Shape (B,)
    :rtype: Tensor
    """
    probs = batched_pdf(x,means,variances=variances,stds=stds)
    return torch.linalg.vecdot(probs,weights[None,:])

## Multiple different gaussians in parallel (pdf can be summed for mixture-of-gaussians)
@torch_fn #coerce inputs to tensors
def batched_logpdf(x:Tensor, means:Tensor|Literal[0], variances:Optional[Tensor]=None, stds:Optional[Tensor]=None)->Tensor: #x: BxD or BxNxD, means: NxD, variances: N or NxDxD, stds: N or NxDxD, result: BxN
    """
    Batch computation of the logarithm of the pdf for N D-dimensional multivariate gaussian distributions at B D-dimensional points x. (You can also specify a pre-expanded tensor with size BxNxD).
    Distributions are parameterized by a sequence of D-dimensional means (NxD tensor) and a sequence of either (co)variances or standard deviations.
    Variance/deviation can be either a sequence of scalars, an N-dimensional tensor, or a sequence of matrixes, an NxDxD tensor.
    The matrix "standard deviation" is the lower triangular matrix which squares to the covariance matrix (that is, S @ S.T = V) whose diagonal entries are positive.
    If the covariances are zero but each dimension has unique variances, the "standard deviation" matrix has a diagonal of the standard deviations for each dimension. 
    See torch.distributions.MultivariateNormal's scale_tril parameter for more details.

    For scalar-valued variances/stds, providing variances is preferred, whereas lower triangular standard deviation matrices are preferred over covariance matrices
    to avoid a costly (and cpu-only!) matrix decomposition for inversion.

    :param x: Points for which to calculate the log pdf. Shape (B,D) or (B,N,D)
    :type x: Tensor
    :param means: Means of the N D-dimensional gaussians, shape (N,D)
    :type means: Tensor|Literal[0]
    :param variances: (Co)variances of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type variances: Optional[Tensor]
    :param stds: Standard deviations of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type stds: Optional[Tensor]
    :return: logarithmic pdfs for each point in x. Shape (B,N)
    :rtype: Tensor
    """
    if x.ndim not in (2,3):
        raise ValueError(f"x must be a 2D tensor with shape (B,D) [or a 3D tensor with shape (B,N,D)], got {x.shape}")
    if not (variances is None) ^ (stds is None): #exclusive or, exactly one of variances and stds is provided
        raise ValueError("Exactly one of variances and stds must be specified, the other must be None")
    if variances is not None:
        var_shape = variances.shape
    elif stds is not None:
        var_shape = stds.shape
    else:
        raise Exception() #should be unreachable
    
    N = var_shape[0]
    D = x.shape[-1]

    if isinstance(means,torch.types.Number):
        means = torch.full((N,D),means,dtype=x.dtype,device=x.device)
    else:
        if means.ndim != 2:
            raise ValueError(f"means must be a 2D tensor with shape (N,D), got {means.shape}")
        if not means.shape[0] == N:
            raise ValueError(f"Variance/deviation shape {var_shape} incompatible with means shape {means.shape}, first dimension (number of distributions) must match!")
        if not means.shape[-1] == D:
            raise ValueError(f"Data point shape {x.shape} incompatible with means shape {means.shape}, last dimension (dimension of distributions) must match!")

    if x.ndim == 2:
        x = x[:,None,:] #add middle dimension, x = Bx1xD
    x = x.expand(-1,N,D) #Expand middle dimension, new shape BxNxD. This will error if a pre-expanded 3D x tensor is supplied whose shape doesn't match the number of distributions N

    if len(var_shape) == 3: #matrix variance
        if not var_shape[-1] == D:
            raise ValueError(f"Data point shape {x.shape} incompatible with variance/std shape {var_shape}, last dimension (dimension of distributions) must match!")
        
        #TODO: consider computing scale_tril from cholesky_ex [can run on the gpu but doesn't do error checking] rather than letting MultivariateNormal use cholesky() [which is confined to the cpu]?
        
        dist = torch.distributions.MultivariateNormal(means,covariance_matrix=variances if variances is not None else None, scale_tril=stds if stds is not None else None)
        return dist.log_prob(x)
    elif len(var_shape) == 1: #scalar variance
        if variances is None:
            assert stds is not None
            variances = torch.square(stds)
        #compute x - mean, add batch dimension to mean
        dx = x - means[None]
        #scalar sigmas are much easier; no mucking about with inverses or decompositions, just divide
        nums = torch.linalg.vecdot(dx,dx)/variances[None]
        return -1/2*(D*(torch.log(variances*2*torch.pi)) + nums)
    else:
        raise ValueError(f"Invalid variance/std shape {var_shape}; variance/std must be a sequence of scalars (N-tensor) or a sequence of DxD matrices (NxDxD-tensor)")

## Multiple different gaussians in parallel (can be summed for mixture-of-gaussians)
@torch_fn #coerce inputs to tensors
def batched_pdf(x:Tensor, means:Tensor|Literal[0], variances:Optional[Tensor]=None, stds:Optional[Tensor]=None)->Tensor: #x: BxD or BxNxD, means: NxD, variances: N or NxDxD, stds: N or NxDxD, result: BxN
    """
    Batch computation of the pdf for N D-dimensional multivariate gaussian distributions at B D-dimensional points x.
    Distributions are parameterized by a sequence of D-dimensional means (NxD tensor) and a sequence of either (co)variances or standard deviations.
    Variance/deviation can be either a sequence of scalars, an N-dimensional tensor, or a sequence of matrixes, an NxDxD tensor.
    The matrix "standard deviation" is the lower triangular matrix which squares to the covariance matrix (that is, S @ S.T = V) whose diagonal entries are positive.
    If the covariances are zero but each dimension has unique variances, the "standard deviation" matrix has a diagonal of the standard deviations for each dimension. 
    See torch.distributions.MultivariateNormal's scale_tril parameter for more details.

    For scalar-valued variances/stds, providing variances is preferred, whereas lower triangular standard deviation matrices are preferred over covariance matrices
    to avoid a costly (and cpu-only!) matrix decomposition for inversion.

    :param x: Points for which to calculate the pdf. Shape (B,D)
    :type x: Tensor
    :param means: Means of the N D-dimensional gaussians, shape (N,D)
    :type means: Tensor|Literal[0]
    :param variances: (Co)variances of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type variances: Optional[Tensor]
    :param stds: Standard deviations of the N D-dimensional gaussians, shape (N,) or (N,D,D). Exactly one of variances or stds must be specified.
    :type stds: Optional[Tensor]
    :return: pdfs for each point in x. Shape (B,N)
    :rtype: Tensor
    """

    logprob = batched_logpdf(x,means,variances=variances,stds=stds)
    return torch.exp(logprob)


## Single Gaussian, not mixture
@torch_fn #coerce inputs to tensors
def gaussian_logpdf(x:Tensor,mean:Tensor|Literal[0],variance:Optional[Tensor|torch.types.Number]=None,std:Optional[Tensor|torch.types.Number]=None)->Tensor: #x: BxD, mean: D, variance: scalar or DxD, std: scalar or DxD, result: B
    """Computes the logarithm of the pdf for a D-dimensional multivariate gaussian at B D-dimensional points x.

    :param x: Points for which to calculate the pdf. Shape (B,D)
    :type x: Tensor
    :param mean: Center of the D-dimensional gaussian, shape (D,)
    :type mean: Tensor | Literal[0]
    :param variance: Co(variance) of the D-dimensional gaussian, scalar or shape (D,D). Exactl one of variance or std must be specified. defaults to None
    :type variance: Optional[Tensor | torch.types.Number], optional
    :param std: Standard deviation of the N D-dimensional gaussians, scalar or shape (D,D). Exactly one of variance or std must be specified. defaults to None
    :type std: Optional[Tensor | torch.types.Number], optional
    :return: logarithm of the pdf for each point in x. Shape (B,)
    :rtype: Tensor
    """
    
    means = mean[None] if isinstance(mean,Tensor) else [mean] if not isinstance(mean,torch.types.Number) else mean
    variances = variance[None] if isinstance(variance,Tensor) else [variance] if variance is not None else variance
    stds = std[None] if isinstance(std,Tensor) else [std] if std is not None else std
    
    return batched_logpdf(x,means,variances=variances,stds=stds)[...,0]

## Single Gaussian, not mixture
@torch_fn #coerce inputs to tensors
def gaussian_pdf(x:Tensor,mean:Tensor|Literal[0],variance:Optional[Tensor|torch.types.Number]=None,std:Optional[Tensor|torch.types.Number]=None)->Tensor: #x: BxD, mean: D, variance: scalar or DxD, std: scalar or DxD, result: B
    """Computes the pdf for a D-dimensional multivariate gaussian at B D-dimensional points x.

    :param x: Points for which to calculate the pdf. Shape (B,D)
    :type x: Tensor
    :param mean: Center of the D-dimensional gaussian, shape (D,)
    :type mean: Tensor | Literal[0]
    :param variance: Co(variance) of the D-dimensional gaussian, scalar or shape (D,D). Exactl one of variance or std must be specified. defaults to None
    :type variance: Optional[Tensor | torch.types.Number], optional
    :param std: Standard deviation of the N D-dimensional gaussians, scalar or shape (D,D). Exactly one of variance or std must be specified. defaults to None
    :type std: Optional[Tensor | torch.types.Number], optional
    :return: the pdf for each point in x. Shape (B,)
    :rtype: Tensor
    """

    means = mean[None] if not isinstance(mean,torch.types.Number) else mean #this feels bad
    variances = variance[None] if isinstance(variance,Tensor) else [variance] if variance is not None else variance
    stds = std[None] if isinstance(std,Tensor) else [std] if std is not None else std

    return batched_pdf(x,means,variances=variances,stds=stds)[...,0]


if __name__ == "__main__":
    B = 200
    N = 2
    D = 2
    means = torch.tensor([[-10,-10],[10,10]],dtype=float)#torch.rand((N,D),dtype=float)
    variances = torch.diag_embed(torch.full(means.shape,0.1,dtype=float)) #4 in every axis cause why not
    weights = torch.arange(0,N)+1

    from IPython import embed; embed()