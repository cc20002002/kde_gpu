
# kde_gpu

Kernel density estimator using Nadaraya-Watson with GPU (CUDA)

Author
------
Chen Chen


Running Environment Setup
------------
You have to have cupy installed to be able to use GPU!!
See https://github.com/cupy/cupy


Similar to [scipy.kde_gaussian](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html) and [statsmodels.nonparametric.kernel_density.KDEMultivariateConditional](https://www.statsmodels.org/stable/generated/statsmodels.nonparametric.kernel_density.KDEMultivariateConditional.html), this project implements Nadaraya-Waston kernel density and kernel conditional probability estimator using cuda through cupy. However, it is much faster than CPU version and it maximise the use of GPU memory.

File tree structure:  
     |--- kde_gpu\\  
     ***|--- __version__.py \\  
     ***|--- nadaraya_watson.py \\  
	 ***|--- conditional_probability.py \\  
     *|--- setup.py \\  
     *|--- example.py \\  
     *|--- README.md \\  

1. Install cupy (i.e. numpy equivalent using CUDA GPU acceleration) according to https://docs-cupy.chainer.org/en/stable/install.html#using-pip 

2. Install `kde_gpu` with following command: (Please use `pip3` if the default `python` in your computer is `python2`)

   ```
   $ pip install kde_gpu
   ```
 This command will run  `setup.py` where we specify the dependencies required to run  `kde_gpu`. The dependencies we require are:

           "scipy>=1.0.0",
           "pandas>=0.20.2",

Please note that if the version number of installed package in your machine is lower than the stated version number, `pip` will uninstall your out-of-date package and install the one with version number greater than or equal to the stated one in `setup.py`.

Example
------------

~~~~
"""
@author: chen.chen.adl@gmail.com
"""

#import kernel_smoothing
from scipy import stats
import pandas as pd
import cupy as cp
import numpy as np
import time

#true distribution is an exponential
rv = stats.expon(0,1)
x = rv.rvs(size=10000)

density_real = rv.pdf(x)

#using scipy package
t1=time.time()
kde_scipy=stats.gaussian_kde(x.T,bw_method='silverman')
kde_scipy=kde_scipy(x.T)
print(time.time()-t1)

#use kde_gpu package
t1=time.time()
kde_cupy=kde(cp.asarray(x.T),bw_method='silverman')
print(time.time()-t1)

df = pd.DataFrame({'x1':x,'kde_scipy':kde_scipy,
                   'kde_cupy':cp.asnumpy(kde_cupy).squeeze(),'real density':density_real})

df['scipy_mean_absolute_error']=np.abs(df['kde_scipy']-df['real density'])
df['cupy_mean_absolute_error']=np.abs(df['kde_cupy']-df['real density'])
print(df.mean())


#true x is truncated normal and true y conditional on x is a uniform distribution, which is independent of x.
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
~~~~
