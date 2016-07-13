# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 21:12:14 2015

@author: kuanweic
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import pandas as pd
import scipy.stats as scs
import statsmodels.api as sm

'''Call Option'''
K=8000
S=np.linspace(7000,9000,100)
h=np.maximum(S-K,0)
plt.figure()
plt.plot(S,h,lw=2.5)

'''Sell Call'''
plt.figure()
h=np.minimum(K-S,0)
plt.plot(S,h,lw=2)



'''Put Option'''
plt.figure()
h=np.maximum(K-S,0)
plt.plot(S,h,lw=2)


'''Sell Put Option'''
plt.figure()
h=np.minimum(S-K,0)
plt.plot(S,h,lw=2)


'''PDF of Normal  random variable'''

def dN(x, mu,sigma):
    z=(x-mu)/sigma
    pdf=np.exp(-0.5*z**2)/math.sqrt(2*math.pi*sigma**2)
    return pdf
    
def simulate_gbm():
    s0=100.0
    T=10.0
    r=0.05
    vol=0.2
    
    np.random.seed(250000)
    gbm_dates=pd.DatetimeIndex(start='30-09-2004',end='30-09-2014',freq='B')
    I=1    
    M=len(gbm_dates)
    dt=1/252
    df=math.exp(-r*dt)
    rand=np.random.standard_normal((M,I))
    s=np.zeros_like(rand)
    s[0]=s0
    for t in range(1,M):
        s[t]=s[t-1]*np.exp((r-vol**2/2)*dt+vol*rand[t]*math.sqrt(dt))
        

pd.DataFrame([10,20,30,40],columns=['number'],index=['a','b','c','d'])