import sys, os, math
import numpy as np
import scipy.interpolate as sint
import scipy.optimize
import bioen.optimize as bop

def adapt_q_range(Iq, expIq):
    """ 
    Generates new intensity array with q-values determined by other (experimental) intensity. Linear interpolation is used.
    """
    IqPrime = expIq.copy()
    IqInt = sint.interp1d(Iq[:, 0], Iq[:, 1])
    for i in range(len(expIq)):
        IqPrime[i, 1] = IqInt(expIq[i, 0])
    #np.savetxt("IqPrime.dat", IqPrime, fmt='%.3f %.6e', delimiter=' ')
    return IqPrime

def get_y_ave(w, y):
    """
    A helper function to be more transparant. 
    """
    return bop.common.getAve(w, y)

def fit_to_ave(w, y, Iq_exp):
    """
    Calculating the average of the calculated scattering intensities and fitting of this average to the experiment. 
    We use two fit paramters: A scale parameter 'a' and an offset parameter 'b'. I.e., I'(q)=a*I(q)+b. 
    """
    y_ave=get_y_ave(w, y)
    f_y_ave = sint.interp1d(Iq_exp[:,0], y_ave)
    def f(q, a, b):
        return (a*f_y_ave(q))+b
    popt, pcov = scipy.optimize.curve_fit(f, Iq_exp[:,0], Iq_exp[:,1], sigma=Iq_exp[:,2])
    return popt[0], popt[1], f

def update_y(a, b, y, yTilde):
    """
    Return fitted y and YTilde matrices for BioEn refinement. 
    """
    return a*y+b, a*yTilde+b

def KL(w, w0):
    """
    Return Kullback-Leibler divergence for optimal weights 'w' with respect to reference weights 'w0'. 
    """
    ind=np.where(w>0)[0]
    return (w[ind]*np.log(w[ind]/w0[ind])).sum()

def set_theta_list(a,b, theta_fac):
    """
    Return list theta values, which are linear on the logarithmic scale and ordered from large (large confidence) to low (low confidence) values. 
    
    Args:
        a: Smallest power of 10. 
        b: Largest power of 10. 
        theta_fac: Number of points between powers of ten. 
    """
    num=theta_fac*(b-a)+1
    theta_list=10**np.linspace(a, b, num)[::-1]
    return theta_list 
