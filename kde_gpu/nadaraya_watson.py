#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:34:13 2019

@author: chen chen
chen.chen.adl@gmail.com

"""
import cupy as cp

def pdf(dataset,bw_method='scott',weight=1):
    #
    # Representation of a kernel-density estimate using Gaussian kernels.
    
    '''
    Nadaraya watson Kernel density estimation is a way to estimate the probability density function (PDF) of a 
    random variable in a non-parametric way. The code currently only works for  uni-variate data. It includes automatic 
    bandwidth determination. The estimation works best 
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
   
    assert dataset.ndim==1
    n=dataset.size
    
    neff = (cp.ones(n)*weight).sum()
    d = 1
    #find band width
    if bw_method=='scott':
        h = neff**(-1./(d+4))
    elif bw_method=='silverman':
        h = (neff * (d + 2) / 4.)**(-1. / (d + 4))
    else:
        h = bw_method
    
    
    dataset = cp.asarray(dataset/h, dtype='float32').T
    dataset = cp.expand_dims(dataset,1)
    XX=cp.broadcast_to(dataset,(n,n))
    XXT=cp.broadcast_to(dataset.T,(n,n))
    norm = cp.absolute(XX-XXT)
    XX = None
    XXT = None
    
    #find k((x-X)/h)
    kxx = cp.copy(norm)
    kxx[norm<1] = 70/81*(1-norm[norm<1]**3)**3
    kxx[norm>=1] = 0
    norm =None
    
    
    kernel = cp.asarray(weight, dtype='float32')
    kernel = cp.broadcast_to(kernel,(n,n))    
    kernel = kxx*kernel
    kde = kernel.mean(0,keepdims=False)/h
    
    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    return kde


