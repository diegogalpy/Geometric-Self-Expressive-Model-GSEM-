import numpy as np
from random import randint

def GSEM(Y, F, l2_reg, l1_reg, alpha):
    """ GSEM implements a multiplicative learning rule to learn the sparse matrix W for the model Y*W.
        This model considers the geometric structure of the points (columns of Y) by the affinity matrix F. 
        input:
            * Y: binary matrix of n x m.
            * F: affinity matrix of m x m.
            * tolX: stopping criteria. It can be set to 1e-2.
            * variance: initial variance for W. It can be set to 0.01.
            * l2_reg: l2 regularization parameter for W. 
            * l1_reg: l1 regularization parameter for W.
            * alpha: regularization parameter for the graph F.
            * gamma: null-diagonal regularization parameter penalty. It can be set to 1e4.
       output:           
            * Yhat: scores for Y.

        author: diego galeano
        date: 03-08-2019
        email: dgaleano@ing.una.py
    """
    tolX = 1e-3    # stopping criteria
    maxiters = 1000 # usually converges before very fast (< 100 iterations) even in big datasets
    gamma = 1e4
    variance = 0.01   
    
    # Get the dimensions
    (n,m) = Y.shape
    
    # initialization   
    W = np.random.uniform(0,np.sqrt(variance),(m,m))   
    W0 = W;
    
    # Diagonal matrix for the graph Laplacian
    D = np.diag(F.sum(1))
    
    # the data covariance
    C = np.dot(Y.transpose(), Y)
       
    # Identity
    I = np.identity(m);
       
    #get machine precision eps
    epsilon = np.finfo(float).eps  
    sqrteps = np.sqrt(epsilon);
    
    for iter in range(maxiters):
        numer = C + alpha*np.dot(W,F)
        denom = np.dot(C,W) + alpha*np.dot(W,D) + l1_reg + epsilon + gamma*I       
       
        W = np.multiply(W, np.divide(numer, denom))
        
        # Delete negative values due to machine precision.
        W.clip(min = 0)
        
        # Get the max change in W      
        dw = np.amax(np.abs(W-W0))/(sqrteps + np.amax(np.abs(W0)));
        
        if dw <= tolX:
            print('Iter', iter, 'dw', dw)
            break
        
        W0 = W;
    
    Yhat = np.dot(Y,W)
    return W    