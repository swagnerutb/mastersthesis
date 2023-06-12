import numpy as np
import algopy

def vanilla_call(X, K, idx=None):
    """max(X - K, 0)"""
    if idx is None:
        if isinstance(X,algopy.utpm.utpm.UTPM):
            return algopy.UTPM.maximum(X-K, X*0)
        else:
            X = X - K
            X[np.where(X > 0)] *= 0
            return X
    else:
        return algopy.UTPM.sum(X[:,idx] - K, axis=1)

def vanilla_call_put(X, call_K=0, put_K=0, call_idx=[], put_idx=[]):
    """Vanilla call and put payoff"""
    if len(put_idx) == 0:
        call_val = algopy.UTPM.sum(algopy.UTPM.maximum(X[:,call_idx]-call_K,
                                                       X[:,call_idx]*0), axis=1)
        return call_val
    
    if len(call_idx) == 0:
        put_val = algopy.UTPM.sum(algopy.UTPM.maximum(put_K - X[:,put_idx],
                                                      X[:,put_idx]*0), axis=1)
        return put_val
    
    call_val = algopy.UTPM.sum(algopy.UTPM.maximum(X[:,call_idx]-call_K,
                                                   X[:,call_idx]*0), axis=1)
    put_val = algopy.UTPM.sum(algopy.UTPM.maximum(put_K - X[:,put_idx],
                                                  X[:,put_idx]*0), axis=1)
    return algopy.UTPM.add(call_val,put_val)

def FX_forward(X, idx_long, idx_short):
    if len(idx_long) != len(idx_short):
        raise Exception('Equal lengths of long and short indeces required')
    if isinstance(X,algopy.utpm.utpm.UTPM):
        return algopy.UTPM.sum(X[:,idx_long] - X[:,idx_short], axis=1)
    else:
        return np.sum(X[:,idx_long] - X[:,idx_short], axis=1)

