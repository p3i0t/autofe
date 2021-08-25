import torch

# use inplace-operation as much as possible to save memory.

#########################
# element-wise functions
#########################

def _protected_log(x: torch.Tensor) -> torch.Tensor:
    """Compute the protected logarithm, valid for both positive and negative values.
    """  
    return  torch.where(x > 0, (x+1).log(), -(-x+1.0).log())


def _protected_sqrt(x: torch.Tensor) -> torch.Tensor:
    """Compute the protected sqrt, valid for both positive and negative values.
    """    
    return torch.where(x > 0, x.sqrt(), -(-x).sqrt())


def _inverse(x: torch.Tensor) -> torch.Tensor:
    """Compute the element-wise inverse. 
    """  
    return 1.0 / x

def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Compute the element-wise sigmoid.
    """ 
    return x.sigmoid_()


def _tanh(x: torch.Tensor) -> torch.Tensor:
    """Compute the element-wise tanh.
    """ 
    return x.tanh_()
    
def _negative(x: torch.Tensor) -> torch.Tensor:
    """Compute the element-wise negative.
    """ 
    return x.neg_()

def _signedpower(x: torch.Tensor, a: float) -> torch.Tensor:
    """Compute the signed power to some scalar `a`.

    Parameters
    ----------
    x : torch.Tensor
        3D Tensor (datetime-like rolling dimension, factor, symbol), factor dim is 1 by default.
    a : float
        scalar of power.

    Returns
    -------
    torch.Tensor
        output.
    """  
    # validate_input(x)  
    return x.sign() * x.abs() ** a
