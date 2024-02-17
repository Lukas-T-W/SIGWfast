# This file is part of SIGWfast.

# SIGWfast is free software: you can use, copy, modify, merge, publish and
# distribute, sublicense, and/or sell copies of it, and to permit persons to
# whom it is furnished to do so it under the terms of the MIT License.

# SIGWfast is distributed in the hope that it will be useful,
# but WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT 
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
# PURPOSE AND NONINFRINGEMENT. See the MIT License for more details.

# You should have received a copy of the MIT License along with SIGWfast. If
# not, see <https://spdx.org/licenses/MIT.html>.

import numpy as np
import scipy.special as special
from math import floor

# For w=1/3 the integration kernels defined below diverge. If w is thus too
# close to w=1/3 we shift it to w=1/3-epsilon. This permits the computation to
# proceed without encountering divergences and does not significantly affect 
# the result. 
def vv(w,epsilon=10**(-4)):
    if abs(w-1/3)<epsilon:
        return 1/3-epsilon
    else:
        return w

# Define beta=(1-3*w)/(1+3*w)
def beta(w):
    w=vv(w)
    beta=(1-3*w)/(1+3*w)
    return beta

# Define the arrays over the integration variables d and s for c_s^2=w.
# d is sampled linearly over its interval [0, 1-epsilon]. The epsilon is
# included so that (d-s) is always at least as large as epsilon.
# The s-array is split into s1array holding values on [1, 1/sqrt(w)-epsilon] 
# and s2array holding values [1/sqrt(w)+epsilon, smax]. These are filled so
# that entries become more frequent as 1/sqrt(w) is approached, where the 
# kernel defined below peaks or even diverges. The bunching of values is 
# controlled by the value of the local variable power, which is set to the 
# default value power=10 here.   
def arrays_w(w,karray,epsilon=10**(-10),nd=100):
    kmin = np.amin(karray)
    power= 10
    w    = vv(w)
    smax = 10/kmin
    ns1  = 200
    if floor(smax)<800:
        ns2 = 600
    else:
        ns2 = floor(600+np.sqrt(smax-800))
    d   =  np.linspace(0,1-epsilon,nd)
    dd1 =  np.repeat(d,ns1)
    dd2 =  np.repeat(d,ns2)
    s1  =  np.flipud(  1/np.sqrt(w)-epsilon \
                     -(1/np.sqrt(w)-1-epsilon)*(np.linspace(0,1,ns1))**(power))
    ss1 =  np.tile(s1,nd)
    s2  =  1/np.sqrt(w)+epsilon \
          +(smax-1/np.sqrt(w)-epsilon)*(np.linspace(0,1,ns2))**(power)
    ss2 =  np.tile(s2,nd)
    return nd, ns1, ns2, d, dd1, dd2, ss1, ss2

# Define arrays for the case c_s^2=1. As the kernels do not have a singularity
# we do not split the s-array. 
def arrays_1(w,karray,epsilon=10**(-10),nd=100):
    kmin = np.amin(karray)
    power= 4
    w    = vv(w)
    smax = 10/kmin
    if floor(smax)<800:
        ns0 = 800
    else:
        ns0 = floor(800+np.sqrt(smax-800))
    d  =  np.linspace(0,1-epsilon,nd)
    dd =  np.repeat(d,ns0)
    s  =  1+epsilon+(smax-1-epsilon)*(np.linspace(0,1,ns0))**(power)
    ss =  np.tile(s,nd)
    return nd, ns0, d, dd, ss

# Define arrays for the case of GWs induced during an era of radiation 
# domination, i.e. c_s^2=w=1/3. These are as arrays_w above, but with w set to
# the fixed value w=1/3.
def arrays_r(karray,epsilon=10**(-10),nd=100):
    kmin = np.amin(karray)
    power= 10
    smax = 10/kmin
    ns1  = 200
    if floor(smax)<800:
        ns2 = 600
    else:
        ns2 = floor(600+np.sqrt(smax-800))
    d   =  np.linspace(0,1-epsilon,nd)
    dd1 =  np.repeat(d,ns1)
    dd2 =  np.repeat(d,ns2)
    s1  =  np.flipud(  np.sqrt(3)-epsilon \
                     -(np.sqrt(3)-1-epsilon)*(np.linspace(0,1,ns1))**(power))
    ss1 =  np.tile(s1,nd)
    s2  =  np.sqrt(3)+epsilon \
          +(smax-np.sqrt(3)-epsilon)*(np.linspace(0,1,ns2))**(power)
    ss2 =  np.tile(s2,nd)
    return nd, ns1, ns2, d, dd1, dd2, ss1, ss2

# Define functions I_J^2 and I_Y^2 as given in arxiv:1912.05583:
    
# Define function I_J^2 for c_s^2=w and s<1/sqrt(w). As this is zero, we do not  
# need to use it, so it is commented out. It is included here for completeness.
#def IJsq1_w(d,s,b):
#    return 0

# Define function I_J^2 for c_s^2=w and s>1/sqrt(w)
def IJsq2_w(d,s,b):
    vv = (1-b)/3/(1+b)
    y  = (s**2+d**2-2/vv)/(s**2-d**2)
    N  = (1+b)**(-2*(1+b))*16**(1+b)/3/vv**2*((2+b)/(3+2*b))**2*(
                                                       special.gamma(b+3/2))**4
    P1 = ((1+y)/(1-y))**(-b/2)/special.gamma(1+b)*special.hyp2f1(
                                                            1+b,-b,1+b,(1-y)/2)
    P2 = ((1+y)/(1-y))**(-b/2)/special.gamma(1+b)*special.hyp2f1(
                                                          3+b,-2-b,1+b,(1-y)/2)
    f  = N/(s**2-d**2)**2*(1-y**2)**b*(P1+(2+b)/(1+b)*P2)**2
    return f

# Define function I_Y^2 for c_s^2=w and s<1/sqrt(w) 
def IYsq1_w(d,s,b):
    vv = (1-b)/3/(1+b)
    y  =-(s**2+d**2-2/vv)/(s**2-d**2)
    N  = (1+b)**(-2*(1+b))*16**(1+b)/3/vv**2*((2+b)/(3+2*b))**2*(
                                                       special.gamma(b+3/2))**4
    Q3 = np.sqrt(np.pi)/2**(1+b)/y/special.gamma(b+3/2)*special.hyp2f1(
                                                            1,1/2,b+3/2,1/y**2)
    Q4 = np.sqrt(np.pi)/2**(3+b)/y**3/special.gamma(b+7/2)*special.hyp2f1(
                                                            2,3/2,b+7/2,1/y**2)
    f  = N/(s**2-d**2)**2*(4/np.pi/np.pi*(Q3+2*(2+b)/(1+b)*Q4)**2)
    return f

# Define function I_Y^2 for c_s^2=w and s>1/sqrt(w) 
def IYsq2_w(d,s,b):
    vv = (1-b)/3/(1+b)
    y  = (s**2+d**2-2/vv)/(s**2-d**2)
    N  = (1+b)**(-2*(1+b))*16**(1+b)/3/vv**2*((2+b)/(3+2*b))**2*(
                                                       special.gamma(b+3/2))**4
    Q1 = np.pi/2/np.sin(-np.pi*b)*(np.cos(-np.pi*b)*((1+y)/(1-y))**(-b/2)/ \
         special.gamma(1+b)*special.hyp2f1(1+b,-b,1+b,(1-y)/2)- \
         special.gamma(1)/special.gamma(1+2*b)*((1-y)/(1+y))**(-b/2)/ \
         special.gamma(1-b)*special.hyp2f1(1+b,-b,1-b,(1-y)/2))
    Q2 = np.pi/2/np.sin(-np.pi*b)*(np.cos(-np.pi*b)*((1+y)/(1-y))**(-b/2)/ \
         special.gamma(1+b)*special.hyp2f1(3+b,-2-b,1+b,(1-y)/2)- \
         special.gamma(3)/special.gamma(3+2*b)*((1-y)/(1+y))**(-b/2)/ \
         special.gamma(1-b)*special.hyp2f1(3+b,-2-b,1-b,(1-y)/2))
    f  = N/(s**2-d**2)**2*(1-y**2)**b*(4/np.pi/np.pi*(Q1+(2+b)/(1+b)*Q2)**2)
    return f

# Define function I_J^2 for c_s^2=1
def IJsq2_1(d,s,b):
    y  = (s**2+d**2-2)/(s**2-d**2)
    N  = (1+b)**(-2*(1+b))*16**(1+b)/3*((2+b)/(3+2*b))**2*(
                                                       special.gamma(b+3/2))**4
    P1 = ((1+y)/(1-y))**(-b/2)/special.gamma(1+b)*special.hyp2f1(
                                                            1+b,-b,1+b,(1-y)/2)
    P2 = ((1+y)/(1-y))**(-b/2)/special.gamma(1+b)*special.hyp2f1(
                                                          3+b,-2-b,1+b,(1-y)/2)
    f  = N/(s**2-d**2)**2*(1-y**2)**b*(P1+(2+b)/(1+b)*P2)**2
    return f

# Define function I_Y^2 for c_s^2=1 
def IYsq2_1(d,s,b):
    y  = (s**2+d**2-2)/(s**2-d**2)
    N  = (1+b)**(-2*(1+b))*16**(1+b)/3*((2+b)/(3+2*b))**2*(
                                                       special.gamma(b+3/2))**4
    Q1 = np.pi/2/np.sin(-np.pi*b)*(np.cos(-np.pi*b)*((1+y)/(1-y))**(-b/2)/ \
         special.gamma(1+b)*special.hyp2f1(1+b,-b,1+b,(1-y)/2)- \
         special.gamma(1)/special.gamma(1+2*b)*((1-y)/(1+y))**(-b/2)/ \
         special.gamma(1-b)*special.hyp2f1(1+b,-b,1-b,(1-y)/2))
    Q2 = np.pi/2/np.sin(-np.pi*b)*(np.cos(-np.pi*b)*((1+y)/(1-y))**(-b/2)/ \
         special.gamma(1+b)*special.hyp2f1(3+b,-2-b,1+b,(1-y)/2)- \
         special.gamma(3)/special.gamma(3+2*b)*((1-y)/(1+y))**(-b/2)/ \
         special.gamma(1-b)*special.hyp2f1(3+b,-2-b,1-b,(1-y)/2))
    f  = N/(s**2-d**2)**2*(1-y**2)**b*(4/np.pi/np.pi*(Q1+(2+b)/(1+b)*Q2)**2)
    return f

# Define the prefactor that multiplies I_J^2 and I_Y^2 to obtain the transfer 
# function.
def pre(d,s):
    y = ((s**2-1)*(d**2-1)/(d**2-s**2))**2
    return y

# Collect result into the integration kernels that multiply the two factors of
# Pofk in the computation of OmegaGW.
# Integration kernel for c_s^2=w and s<1/sqrt(w)
def kernel1_w(d,s,b):
    y = pre(d,s)*IYsq1_w(d,s,b)
    return y

# Integration kernel for c_s^2=w and s>1/sqrt(w)
def kernel2_w(d,s,b):
    y = pre(d,s)*(IJsq2_w(d,s,b)+IYsq2_w(d,s,b))
    return y

# Integration kernel for c_s^2=1
def kernel_1(d,s,b):
    y = pre(d,s)*(IJsq2_1(d,s,b)+IYsq2_1(d,s,b))
    return y

# Integration kernel for radiation domination, c_s^2=w=1/3, for s<sqrt(3)
def kernel1_r(d,s):
    y = 12*(d**2-1)**2*(s**2-1)**2*(d**2+s**2-6)**4/(s**2-d**2)**8*(
        (np.log((3-d**2)/(3-s**2))+2*(s**2-d**2)/(d**2+s**2-6))**2)
    return y

# Integration kernel for radiation domination, c_s^2=w=1/3, for s>sqrt(3)
def kernel2_r(d,s):
    y = 12*(d**2-1)**2*(s**2-1)**2*(d**2+s**2-6)**4/(s**2-d**2)**8*(
        (np.log((3-d**2)/(s**2-3))+2*(s**2-d**2)/(d**2+s**2-6))**2+np.pi**2)
    return y

# Define the factor containing the two instances of P_zeta(k)
def Psquared(d,s,P,k):
    y = P(k/2*(s+d))*P(k/2*(s-d))
    return y

# Define an integrator over 1D arrays based on the trapezoidal rule
def intarray1D(f,dx):
    S = (f[1:]+f[0:-1])*0.5*(dx[1:]-dx[0:-1])
    return np.sum(S)
