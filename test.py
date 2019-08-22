#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 14:29:33 2019

@author: chen chen
chen.chen.adl@gmail.com
"""

from kde_gpu import conditional_probability
from kde_gpu import nadaraya_watson
from scipy import stats
import pandas as pd
import cupy as cp
import numpy as np
import time

rv = stats.expon(0,1)

x = rv.rvs(size=10000)

density_real = rv.pdf(x)

t1=time.time()
kde_scipy=stats.gaussian_kde(x.T,bw_method='silverman')
kde_scipy=kde_scipy(x.T)
print(time.time()-t1)

t1=time.time()
kde_cupy=nadaraya_watson.pdf(cp.asarray(x.T),bw_method='silverman')
print(time.time()-t1)

df = pd.DataFrame({'x1':x,'kde_scipy':kde_scipy,
                   'kde_cupy':cp.asnumpy(kde_cupy).squeeze(),'real density':density_real})

df['scipy_mean_absolute_error']=np.abs(df['kde_scipy']-df['real density'])
df['cupy_mean_absolute_error']=np.abs(df['kde_cupy']-df['real density'])
print(df.mean())


rv = stats.truncnorm(-3,2,30,10)
nsample=10000
x = cp.asarray(rv.rvs(nsample))
ycondx = cp.asarray(cp.random.rand(nsample))
y = 10*(ycondx-0.5)+x

cdf_conditional_real = ycondx
df = pd.DataFrame({'y':cp.asnumpy(y),'x':cp.asnumpy(x),'real density':cp.asnumpy(cdf_conditional_real)})

df['nadaraya watson']= conditional_probability.cdf(y,x)
df['nw_error']=np.abs(df['nadaraya watson']-df['real density'])
df.mean()