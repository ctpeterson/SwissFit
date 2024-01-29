import numpy as _numpy # General numerical operations
import scipy as _scipy

# Calculate square root of inverse of covariance matrix "M"
def cov_inv_SVD(M, square_root = False, return_USV = False, svdcut = None):
    """ 
    Calculate Cov^-1/2:
        
    Here, we calculate the inverse of the covariance matrix by SVD 
    decomposition and take its "square root". Let's write the 
    SVD decomposition of the covariance matrix as 

    Cov = U * S * V^T,

    then the inverse is

    Cov^-1 = V^T * S^-1 * U,

    where [S^-1]_ij = 1/S_ij if i=j and 0 if i != j. We can then
    define the "square root" in terms of the SVD decomposition as

    Cov^-1/2 == S^-1/2 * U,

    so that 

    Cov^-1 = (Cov^-1/2)^T * Cov^-1/2.

    This way of doing things is useful for calculating the residuals,
    which are

    resid == Cov^-1/2 * (y_true - y_predict).

    Moreover, to get the chi^2

    chi^2 = resid^T * resid,

    so all we need to do is store Cov^-1/2.

    Numpy has a weird choice of notation, where V is already 
    transposed, so if the transposes look off, they're not.
    """
    # SVD decomposition
    U, S, V = _numpy.linalg.svd(M, hermitian = True)

    # Check if SVD cut to be applied
    if svdcut is not None: S = _numpy.array([svdcut if s < svdcut else s for s in S])
    
    # Get inverse (M^{-1} or M^{-1/2})
    if square_root: inverse = _numpy.matmul(_numpy.diag(S**-0.5), _numpy.transpose(U))
    else: inverse = _numpy.matmul(
            _numpy.transpose(V), # V^{T}
            _numpy.matmul(
                _numpy.diag(1. / S), # 1/S
                _numpy.transpose(U) # U^{T}
            )
    ) # = V^{T} * 1/S * U^{T} = M^{-1}

    # Return appropriate data
    if return_USV: return U, S, V, inverse
    else: return inverse

# Calculate Moore-Penrose pseudoinverse for parameter covariance estimation
def pinv(M, herm = True):
    """
    NumPy's pinv and SciPy's pinvh perform poorly for the set
    of problems that this code aims to solve.
    """
    return _scipy.linalg.pinv(M, rcond = 1e-64)

# Log determinant
def logdet(M):
    (sgn, logdet_M) = _numpy.linalg.slogdet(M)
    if sgn < 0.: print('Warning! det(fit.cov) < 0!')
    return logdet_M
