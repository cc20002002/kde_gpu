﻿
# kde_gpu

Kernel density estimator using Nadaraya-Watson with GPU (CUDA)

Author
------
Chen Chen


Running Environment Setup
------------

We implemented nadaraya waston kernel density and kernel conditional probability estimator using cuda through cupy. It is much faster than cpu version but it requires GPU with higher memory.

1. Make a folder with name "data" in current directory. Then copy ORL and CroppedYaleB dataset inside. Please make sure you have the following file tree structure:
     |--- nmf \\
     	|--- \__version__.py \\
      	|--- main.py \\
      |--- setup.py \\
      |--- README.md \\

 2. Install `kde_gpu` with following command: (Please use `pip3` if the default `python` in your computer is `python2`)

   ```
   $ pip install -e .
   ```
 This command will run  `setup.py` where we specify the dependencies required to run  `nmf`. The dependencies we require are:

           "cupy>=6.2.0",
           "scipy>=1.0.0",
           "pandas>=0.20.2",

Please note that if the version number of installed package in your machine is lower than the stated version number, `pip` will uninstall your out-of-date package and install the one with version number greater than or equal to the stated one in `setup.py`.

3. 
from scipy import stats
import pandas as pd
import cupy as cp
import numpy as np
import time

#compare speed with scipy package using CPU
rv = stats.expon(0,1)

x = rv.rvs(size=10000)

density_real = rv.pdf(x)

t1=time.time()
kde_scipy=stats.gaussian_kde(x.T,bw_method='silverman')
kde_scipy=kde_scipy(x.T)
print(time.time()-t1)

t1=time.time()
kde_cupy=kde(cp.asarray(x.T),bw_method='silverman')
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

df['nadaraya watson']= kernel_smoothing_ecdf(y,x)
df['nw_error']=np.abs(df['nadaraya watson']-df['real density'])
df.mean()