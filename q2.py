
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 20:18:53 2020

@author: caitlin
"""
import numpy as np
import math as math
import matplotlib as plt
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import os
from uncertainties import ufloat
from uncertainties import unumpy
import scipy.linalg as linalg
from scipy.optimize import curve_fit
from matplotlib.patches import Polygon
from scipy.integrate import quad
import scipy.integrate as integrate
import scipy.special as special
from scipy.integrate import quad
import matplotlib
matplotlib.axes.Axes.errorbar
matplotlib.pyplot.errorbar

def n_o(lambdaa):                       #function defined for no
    n_o=np.sqrt(2.7359+(0.01878/(lambdaa**2-0.01822))-0.01354*lambdaa**2)
    return n_o

def n_e(lambdaa):                       #function defined for ne
    n_e=np.sqrt(2.3753+(0.01224/(lambdaa**2-0.01667))-0.01516*lambdaa**2)
    return n_e

def n_e_theta (lambdaa, theta):         #function defined for angle dependent ne
    n_e_f=n_e(lambdaa)
    n_o_f=n_o(lambdaa)
    n_o_shm=n_o(lambdaa*0.5)
    n_o_e= n_o_f*np.sqrt(((1+(np.tan(theta))**2)/(1+((n_o_f/n_e_f)*np.tan(theta))**2)))
    return n_o_e

#note n_o_f not n_o_shm like previous question
l_p=0.5                                 # pump wavelength

l_s_array=np.arange(0.6, 1, 0.01)       # array with a range of signal wavelength

angle=np.zeros(len(l_s_array))          #empty array to fill with phasematch angle

for i in range(len(l_s_array)):         #looping over signal wavelenth
    
    l_s=l_s_array[i]

    def phase_match(theta):
        l_i=(l_s*l_p)/(l_s-l_p)        #idler wavelength derived from set signal and pump wavelength
        n_p=n_e_theta(l_p,theta)       # pump refractive index, angle dependent extraordinary
        n_s=n_o(l_s)                   # signal refractive wavelength
        n_i=n_o(l_i)                   #idler refractive index
        diff=(n_p/l_p)-(n_s/l_s)-(n_i/l_i)    #pahse matching condition, should equal 0
        return diff
        
    angle[i]=fsolve(phase_match, np.deg2rad(22))  #solving of angle that satisfies phasematching

angle=np.rad2deg(angle)    
    
theta=np.deg2rad(22)     #now finding wavelength for phasematchning at theta=22

def find_ls(l_s):
    l_p=0.5
    n_p=n_e_theta(l_p,theta)  # pump refractive index, angle dependent ne
    l_i=(l_s*l_p)/(l_s-l_p)   # idler wavelength given set pump and signal wavelength
    eq=(n_p/l_p)-((n_o(l_s))/l_s)-((n_o(l_i))/(l_i))  #phasematching to find wavelength where phasematching 
    return eq                                         #angle is 22

l_s=fsolve(find_ls, 0.6)      #solving for the signal wavelength

l_i=(l_s*l_p)/(l_s-l_p)       #find idler wavelength given signal and pump wavelengths

print("at 22o, lambda signal=",l_s,"  lambda idel=", l_i)

plt.plot(angle, l_s_array, label=r"$\lambda$, signal",color="steelblue") #plot of signal wavelength as function of angle
l_i_array=(l_s_array*l_p)/(l_s_array-l_p)
plt.plot(angle, l_i_array, label=r"$\lambda$, idler",color="darkorange") #plot of idler wavelength as function of angle
plt.xlabel(r'Angle for phase-matching, $\theta$')
plt.ylabel(r"Wavelength, $\mu$m")
plt.plot([min(angle), max(angle)], [l_s,l_s], linestyle="dotted", color="steelblue")   #signal wavelength at phasematching angle of 22 degrees
plt.plot([min(angle), max(angle)], [l_i,l_i], linestyle="dotted", color="darkorange")  #idler wavelength at phasematching angle of 22 degrees
plt.plot([min(angle), max(angle)], [l_p,l_p], color="forestgreen", label=r"$\lambda$, pump") #pump wavelength at phasematching angle of 22 degrees

plt.plot([22,22],[0.45,max(l_i_array)], linestyle="dotted", color="gray", label=r"$\theta=22^o$")  #22 degrees vertical line
plt.plot(max(angle),1,'o',color="tomato",label="Reverse SHG")   #SHG point
plt.grid(linewidth=0.5)
plt.legend()


#check as ls and li colours swap for curves and horizontal lines
print(max(angle))  #print phase match angle







