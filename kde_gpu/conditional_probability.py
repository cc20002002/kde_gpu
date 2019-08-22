#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:34:13 2019

@author: chen chen
chen.chen.adl@gmail.com

"""

import cupy as cp

def cdf(y,x,bw_method='scott',weight=1):
    '''
    Nadaraya watson conditional probability estimation is a way to estimate the conditional probability of a 
    random variable y given random variable x in a non-parametric way. It works for both uni-variate and 
    multi-variate data. It includes automatic bandwidth determination. The estimation works best 
    for a unimodal distribution; bimodal or multi-modal distributions tend to be oversmoothed.
    Parameters
    dataset: array_like
    Datapoints to estimate from. Currently, it only supports 1-D array.
    
    bw_method:str, scalar or callable, optional
    The method used to calculate the estimator bandwidth. 
    This can be ‘scott’, ‘silverman’, a scalar constant. 
    If a scalar, this will be used directly as kde.factor. 
    If None (default), ‘scott’ is used. See Notes for more details.
    
    weights:array_like, optional
    weights of datapoints. This must be the same shape as dataset. 
    If None (default), the samples are assumed to be equally weighted
    '''
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    assert (x.ndim==1) & (y.ndim==1)
    NN = y.size
    d = 1
    neff = (cp.ones(NN)*weight).sum()
    if bw_method=='scott':
        h = neff**(-1./(d+4))
    elif bw_method=='silverman':
        h = (neff * (d + 2) / 4.)**(-1. / (d + 4))
    else:
        h = bw_method

    x = x.reshape((-1,1))    
    x = cp.asarray(x/h, dtype='float32')
    y = cp.asarray(y, dtype='float32')
    XX=cp.broadcast_to(x,(NN,NN))
    XXT=cp.broadcast_to(x.T,(NN,NN))
    xx = cp.absolute(XX-XXT)
    
    XX = None
    XXT = None
    xx2 = cp.copy(xx)
    xx[xx2<1] = 70/81*(1-xx[xx<1]**3)**3
    xx[xx2>=1] = 0
    xx2 =None
    
    y = y.reshape((-1,1))
    yy = y<=y.T
    kernel = cp.asarray(weight, dtype='float32')
    kernel = cp.broadcast_to(kernel,(NN,NN))
    kernel = xx*kernel
    weight = kernel/kernel.sum(0,keepdims =True)
    cdf = (weight*yy).sum(0,keepdims =True).T
    #cv = cp.asnumpy((((yy-cdf)/(1-weight))**2*kk).mean())
    weight = None
    kernel = None
    yy = None
    cdf2 = cp.asnumpy(cdf)
    cdf = None
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return cdf2



