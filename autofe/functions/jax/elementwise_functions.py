import jax
import jax.ops
import jax.numpy as jnp



#########################
# element-wise functions
#########################

def _replace_0_to_nan(x):
    return jnp.where(x==0.0, float('nan'), x)


def _replace_inf_to_nan(x):
    return jnp.where(jnp.abs(x)==float('inf'), float('nan'), x)


def _protected_log(x):
    """Compute the protected logarithm, valid for both positive and negative values.
    """  
    return  jnp.where(x > 0, jnp.log(x+1), -jnp.log(-x+1.0))


def _protected_sqrt(x):
    """Compute the protected sqrt, valid for both positive and negative values.
    """    
    return jnp.where(x > 0, jnp.sqrt(x), -jnp.sqrt(-x))


def _inverse(x):
    """Compute the element-wise inverse. 
    """  
    return 1.0 / x

def _sigmoid(x):
    """Compute the element-wise sigmoid.
    """ 
    return 1 / (1 + jnp.exp(-x))


def _tanh(x):
    """Compute the element-wise tanh.
    """ 
    return jnp.tanh(x)
    
def _negative(x):
    """Compute the element-wise negative.
    """ 
    return -x

def _signedpower(x, a: float):
    """Compute the signed power to some scalar `a`.
    """  
    # validate_ijnput(x)  
    return jnp.sign(x) * jnp.abs(x) ** a
