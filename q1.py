
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

lambdas = np.arange(0.5, 2.5, 0.01)   #range of wavelengths

theta=np.zeros(len(lambdas))        #empty array for phase match angle
n_e=np.zeros(len(lambdas))          #as above for ordinary index
n_o=np.zeros(len(lambdas))          #as above for extraordinary index    

for i in range(len(lambdas)):      #loop around all wavelengths in array above
    lambda_f=lambdas[i]
    lambda_shm=lambda_f/2          # SHG have 0.5 of the original wavelength
    n_o_f=np.sqrt(2.7359+(0.01878/(lambda_f**2-0.01822))-0.01354*lambda_f**2)   #sellmeier for fundimental
    n_e_f=np.sqrt(2.3753+(0.01224/(lambda_f**2-0.01667))-0.01516*lambda_f**2)
    
    n_o_shm=np.sqrt(2.7359+(0.01878/(lambda_shm**2-0.01822))-0.01354*lambda_shm**2)   #sellmeier for fundimental
    n_e_shm=np.sqrt(2.3753+(0.01224/(lambda_shm**2-0.01667))-0.01516*lambda_shm**2)

    
    def diff_in_RI(theta):
        n_e_theta=n_o_shm*np.sqrt(((1+(np.tan(theta))**2)/(1+((n_o_shm/n_e_shm)*np.tan(theta))**2)))
        difference=n_e_theta-n_o_f   #phase matching condition, should be zero
        return difference
    
    solution=fsolve(diff_in_RI, 0.4)  #solving for phasematching condition
    theta[i]= np.rad2deg(solution)    #converting to degrees from radians
    
    n_e[i]=n_e_f                      # add loop values to an array to call outside of loop
    n_o[i]=n_o_f                      #as above

plt.figure(1)
plt.plot(lambdas, theta)              #plot of phase matching angle for each wavelength
plt.xlabel(r"Wavelength, $\mu$m")
plt.ylabel(r"Angle, $\theta$")
plt.title(r"Phase matching angle for $\lambda$=0.5-2.5 $\mu$m")
plt.grid()

plt.figure(2)
plt.plot(lambdas,n_o, label=r"$n_o$") #plot of sellmeier for fundamental wavelengths
plt.plot(lambdas,n_e, label=r"$n_e$")
plt.xlabel(r"Wavelength, $\mu$m")
plt.ylabel("Refractive index, n")
plt.title(r"Fundimental $n_o$ and $n_e$ from Sellmeier")
plt.legend()
plt.grid()


#%% for 1um specifically


lambda_f=1
lambda_shm=lambda_f/2   #SHM is half fundamental wavelength

n_o_f=np.sqrt(2.7359+(0.01878/(lambda_f**2-0.01822))-0.01354*lambda_f**2)
n_e_f=np.sqrt(2.3753+(0.01224/(lambda_f**2-0.01667))-0.01516*lambda_f**2)
n_o_shm=np.sqrt(2.7359+(0.01878/(lambda_shm**2-0.01822))-0.01354*lambda_shm**2)
n_e_shm=np.sqrt(2.3753+(0.01224/(lambda_shm**2-0.01667))-0.01516*lambda_shm**2)


def phase_match(theta):
    n_e_theta=n_o_shm*np.sqrt(((1+(np.tan(theta))**2)/(1+((n_o_shm/n_e_shm)*np.tan(theta))**2)))
    return n_e_theta-n_o_f  #difference between these two is zero Phasematching condition
      
angle=fsolve(phase_match, 0.3)      # solving for angle where phasematching is zero       
print("For 1um wave, phase matching occurs at ", np.rad2deg(angle[0]), "degrees")

theta=np.arange(0,np.pi,0.0001)     # angle array, 0 to 180 degrees
n_e_theta=n_o_shm*np.sqrt(((1+(np.tan(theta))**2)/(1+((n_o_f/n_e_f)*np.tan(theta))**2)))
                                    # Refractive index for extraordinary ray of SHM for a range of angles

plt.figure(3)
plt.plot(np.rad2deg(theta), n_e_theta, label=r"$n_{e,2\omega}$")   #plot of ne for SHG to observe overlap-angle dependent
plt.plot([0, 180],[n_o_f, n_o_f], label=r"$n_{o,\omega}$")         # straight line showing invariant no for fundamental
plt.plot([np.rad2deg(angle[0]),np.rad2deg(angle[0])],[1.56, n_o_f], linestyle="dashed", color="grey",label=r"$\Delta \beta=0$") #straight line showing where phase matching condition is satisfied..
plt.xlabel(r"Phase-matching angle, $\theta$")
plt.ylabel("Refractive Index, n")
plt.title(r"Refractive indices for 1 $\mu$m fundimental")
plt.legend()




#%%


