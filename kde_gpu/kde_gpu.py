#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 14:34:13 2019

@author: chen chen
chen.chen.adl@gmail.com

"""
import pyreadr
import pandas as pd
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
#import statsmodels.api as sm
#from IPython import embed

#import tensorflow as tf
#import tensorflow_probability as tfp

#from scipy.stats import t


def kde(dataset,bw_method='scott',weight=1):
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

#boundary correction
def kernel_smoothing_ecdf(y,x,bw_method='scott',weight=1):
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


#boundary correction
def kernel_smoothing_ecdf_weighted(y,x,dampmin=1e-30,maxit=500,lam=0,bw_method='scott',weight=1):
    
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
    NN = x.size  
    
    x = x.reshape((-1,1))
    x = cp.asarray(x/h, dtype='float32')
    y = cp.asarray(y, dtype='float32')
    XX=cp.broadcast_to(x,(NN,NN))
    XXT=cp.broadcast_to(x.T,(NN,NN))
    xx = XX-XXT    
    XX = None
    XXT = None
    #print(mempool.used_bytes())
    kxx = cp.absolute(xx, dtype='float32')
    kxx[kxx<1] = 70/81*(1-kxx[kxx<1]**3)**3
    kxx[cp.absolute(xx, dtype='float32')>=1] = 0
    xx = xx*kxx
    kernel = cp.asarray(weight, dtype='float32')#weight
    kernel = cp.broadcast_to(kernel,(NN,NN))
    
    #Levenberg Marquardt
    whileii=0
    #lam = -1/(xx.max(0)+xx.max(0).mean())/2
    lam = cp.zeros(xx.shape[0],dtype='float32')#-1/(xx.max(0))/2
    max_change = 1
    residual_rhs = 1e10
    damp = 1e-2
    # Levenberg Marquardt method of finding better weighting for adjusted Nadaraya waston
    while ((max_change>2e-100) | (residual_rhs>1e-100)) & (whileii<maxit):
        whileii=whileii+1
        lam2 = cp.broadcast_to(lam,(NN,NN))
        dpt_constraint = cp.asarray(xx/(1+lam2*xx), dtype='float64')
        lam2 = None
        ddpt_constraint = -dpt_constraint**2
        ddpt_constraint = (kernel*ddpt_constraint).sum(0)
        dpt_constraint = (kernel*dpt_constraint).sum(0)
        residual_rhs_old = residual_rhs
        residual_rhs =  cp.absolute(dpt_constraint).mean() #calculate residual
        change = dpt_constraint*ddpt_constraint/(ddpt_constraint**2+damp)
        max_change = cp.absolute(change).max()
        #mempool.free_all_blocks()
        #print(pinned_mempool.n_free_blocks())
        #pinned_mempool.free_all_blocks()        
        dpt_constraint = None
        ddpt_constraint = None
   
        '''
        lam2 = cp.broadcast_to(lam,(NN,NN))
        lam2 = cp.logical_not(((1+lam2*xx)>=0).prod(0))
        #lam2 = None
        lam[lam2] = lam[lam2]/100
        
        if cp.any(lam>0):
            lam[lam>0] = -cp.random.rand(int((lam>0).sum()))/(xx[:,lam>0].max(0))
        #lam = cp.maximum(-1/(xx+1e-4),lam)
        #obj = cp.log(1+lam*xx+1e-4).sum()
        '''
        
        #print(max_change)
        #print(residual_rhs)
        if (residual_rhs_old>=residual_rhs):
            lam = lam - change   
            if ((whileii%20)==0): print(max_change,' ',residual_rhs,' ',damp,' ',lam.max(),lam.min(),' any NA ',cp.isnan(change).any())
            if (damp>dampmin): damp = damp/2
            #damp = damp/2
            change = None
            
        elif (residual_rhs_old<residual_rhs) :
            damp = damp*4
        #print(damp)               
        #print(obj)
    residual_rhs = None   
   
    
    p =  1/(1+lam*xx)*kernel
    p = cp.asarray(p,dtype='float64')
    p = p/p.sum(0)
    if cp.any(p<-1e-3): 
        print('kernel smoothing weighting is not converging in finding outlier, should be all positive')
    p[p<0] = 0
    p = p/p.sum(0)
    
    
    kernel = cp.asarray(kxx* p,dtype='float32')    

    print(lam.max(),lam.min(),p.max(),p.min())
    print('this should be zero. actual residual:',cp.absolute((xx*p).sum(0)).max())
    print('sum of probability should be 1, so this should be 0. Actual residual:',cp.absolute(sum(p)-1).mean())
    
    xx = None
    lam = None   
        
    
    
    
 
    kxx = cp.asarray(kxx * p,dtype='float32')
    #xx2 =None
    p = None
    

    kernel = kxx*kernel
    kernel_de = cp.broadcast_to(kernel.sum(0,keepdims =True),(NN,NN))
    
    y = y.reshape((-1,1))
    yy = y<=y.T
    weight = kernel/kernel_de
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
