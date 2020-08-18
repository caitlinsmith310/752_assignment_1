#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 19:57:22 2020

@author: caitlin
"""


import numpy as np
import math as math
import matplotlib as plt
from scipy.optimize import fsolve
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from uncertainties import ufloat
from uncertainties import unumpy
import scipy.linalg as linalg
from scipy.optimize import curve_fit
from matplotlib.patches import Polygon
from scipy.integrate import quad
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import quad
import matplotlib
matplotlib.axes.Axes.errorbar
matplotlib.pyplot.errorbar

colour=np.array(["#1f77b4","#ff7f0e","#2ca02c","#d62728", "#9467bd", "#8c564b",
                 "#1f77b4","#ff7f0e","#2ca02c","#d62728", "#9467bd"," #1f77b4",
                 "#ff7f0e","#2ca02c","#d62728", "#9467bd",])

             
T_0=1   # pulse width in ps
P_0=1   #initial max power in watts


j=complex(0,1)                  #defining imaginary symbol

window=40*T_0                   #width of times for pulse to be studied
points=2**15                    #number of intervals in the window
dt=window/points                #temporal resolution
T=np.arange(-window/2,window/2,dt)    #time steps array

Omega=2*np.pi*np.fft.fftfreq(len(T),dt)   #frequency array



def split_step(A,h, gamma, beta2):       #split step definition
    a=A*np.exp(j*h*gamma*abs(A)**2)        
    step1=np.fft.fft(a)
    b=np.fft.ifft(np.exp((j*h*beta2*Omega**2)/2)*step1)
    return b

def Gauss_peak(x, A, d):
    y = A*np.exp(-x**2/(2*d**2))   #Defining a function for one Gaussian peak
    return y 



A_0=np.sqrt(P_0)*np.exp(-T**2/(2*T_0**2))   #initial pulse shape for gaussian

#%% DISPERSION ONLY
beta2=20 ##ps**2/km
gamma=0 ##1/w*km

L_D=T_0**2/abs(beta2)   #dispersion length

increments=2            #defined how many L_D are in the h step size
h=increments*L_D   
z=0                     #reset distance to zero

repeats=4               #number of times to move h steps

p=np.zeros(2)           #empty matrix for gaussian widht/amplitude guess
p[0]=1                  #inital peak height from curvefit
p[1]=0.707              #initial peak width from curvefit

A=A_0                   #reset pulse to be the inital pulse shape

for n in range (repeats+1):
    
    pguess = [p[0],p[1]]       #Our guess is made by varying parameters and observing the effect on the theoretical curve 
    p,cov = curve_fit(Gauss_peak,T,(abs(A))**2, p0=pguess)   #guess gaussian parameters
    yestimate = Gauss_peak(T , *p)  
    print("At ", round(z/L_D,2),"L_d, Width:", p[1],"+/-", np.sqrt(cov[1][1]))
    print("|A(z,t)|^2", p[0],"+/-", np.sqrt(cov[0][0]))  #print width of pulse

    plt.figure(1)
    plt.plot(T, abs(A)**2, label=round(z*increments/h,2), color=colour[n]) #plot envelope
    #plt.plot(T, (A**2*np.exp(-j*2*T)).real, color=colour[n])              #plot chirp
    #plt.plot([-p[1],p[1]],[0.5*max(abs(A)**2),0.5*max(abs(A)**2)])        #plot pulse FWFM
    plt.legend(title=r"Distance $L_D$ travelled")
    plt.ylabel(r"$|A(z,T)|^2, W$")
    plt.xlabel("T, ps")
    #plt.xlim(-50,50)
    plt.title(r"Temporal Domain, $\beta_2$ =" +str(beta2)+ r"ps$^2$/km ")
    
    plt.figure(2)
    FA=abs(np.fft.fft(A))**2                                                    #freq power given by magntude squared of fft of A
    plt.plot(Omega, FA,label=round(z*increments/h,2), color=colour[n])          #plot of freq spectrum across the range of frequencies
    plt.legend(title=r"Distance $L_D$ travelled")
    plt.ylabel(r"$|\tilde[A](z,T)|^2$, W")
    plt.xlabel(r"$\Omega$, THz")
    plt.xlim(-10, 10) 
    plt.title(r"Frequency Domain, $\beta_2$ =" +str(beta2)+ r"ps$^2$/km ")
    
    if n==0:                          #plot chirp for Z that are not 0
        plt.figure(3)
        plt.plot(T,beta2*T, label=r"+$\beta_2$")
        plt.plot(T,-beta2*T, label=r"-$\beta_2$")
        #plt.plot(T,np.angle(A))
        plt.title(r"Chirp, $\beta_2$ =$\pm$" +str(beta2)+ r"ps$^2$/km ")
        plt.legend()
        plt.ylabel(r"$\delta\omega(T)$")
        plt.xlabel("T, ps")
        
    A=split_step(A,h,gamma,beta2)   #looping, acting split step on new A each time
    z+=h                            #add h each loop to get total distance propagated.
    
print("distance travelled:", 1000*z,"m")

#%%  Non linearity only          
             
beta2=0 ##ps**2/km
gamma=-5 ##1/w*km

L_NL=abs(1/(gamma*P_0))         #nonlinear length
print(r"$L_{NL}$=",L_NL,"km")

increments=1               #number of nolinear lengths per step size
h=increments*abs(L_NL)    

repeats=3                   #number of loops

A=A_0                       #reset values to initial terms
z=0

for n in range (repeats+1):
    
    plt.figure(4)
    plt.plot(T, abs(A)**2, label=round(z*increments/h,2), color=colour[n])   #temporal power envelope
    #plt.plot(T, (A**2*np.exp(-j*10*T)).real, color=colour[n])               #chir[]
    plt.legend(title=r"$L_{NL}$ travelled")
    plt.ylabel(r"$|A(z,T)|^2, W$")
    plt.xlabel("T, ps")
    plt.xlim(-5, 5) 
    plt.title(r"Temporal Domain, $\gamma$="+str(gamma)+r" W$^{-1}$km$^{-1}$")
    
    plt.figure(5)
    FA=abs(np.fft.fft(A))**2                                                   #Freq of pulse, abs squared value of fft(A)                        
    plt.plot(Omega, FA,label=round(z*increments/h,2), color=colour[n])         #plot freq envelope
    plt.legend(title=r"$L_{NL}$ travelled")
    plt.ylabel(r"$|\tilde[AL_NL](z,T)|^2$, W")
    plt.xlabel(r"$\Omega$, THz")
    plt.xlim(-10, 10) 
    plt.title(r"Frequency Domain, $\gamma$="+str(gamma)+r" W$^{-1}$km$^{-1}$")
    
    plt.figure(6)
    dpodt=np.gradient(abs(A)**2, dt)   #d power over d time
    chirp=-gamma*z*dpodt              #definition from notes
    plt.plot(T, chirp,  label=round(z*increments/h,2), color=colour[n])
    plt.title(r"Chirp, $\gamma$="+str(gamma)+r" W$^{-1}$km$^{-1}$")
    plt.ylabel(r"$\delta\omega(T)$")
    plt.legend(title=r"$L_{NL}$ travelled")
    plt.xlim(-5, 5) 
    plt.xlabel("T, ps")
        
    A=split_step(A, h, gamma, beta2)  #applying split step every loop
    z+=h                              #add h after each loop to work out total distance travelled
    
print("distance travelled:", ((z-h)),"km")


#%%% Soliton
P_0=1       
T_0=1
A_0=np.sqrt(P_0)/(np.cosh((T/T_0)))   #sech pulse

beta2=-20
     
L_D=T_0**2/abs(beta2)                #set dispersion length
gamma=1/(P_0*L_D)                    #define gamma so L_D=L_NL

L_NL=1/(gamma*P_0)

print("Characteristic lengths:",L_D, L_NL)  #check lengths are the same

increments=0.05    #h step is 5\% of L_D, not needed as solition is invarient, commutative, operators constant over z
h=increments*L_D   #ensuring h is less than 5% of the smallest Lenghts scale

A=A_0             #reset A to unpropagated form
z=0

repeats=80        #number of repeats of loop
plot_numbers=[0,0.25*repeats,0.5*repeats,0.75*repeats,repeats] #only only plots a few pulses
  
for n in range (repeats+1):

    if n in plot_numbers:
        plt.figure(7)
        plt.plot(T, abs(A)**2, label=round(z*increments/h,2))    #pulse envelope
        #plt.plot(T, (A**2*np.exp(-j*10*T)).real, label=round(z*increments/h,2))#, color=colour[n])  #chirp
        plt.legend(title=r"Distance $L_D$ travelled")
        plt.ylabel(r"$|A(z,T)|^2, W$")
        plt.xlabel("T, ps")
        plt.xlim(-5,5)
        plt.title(r"Temporal Domain, $\beta_2$ =" +str(beta2)+ r"ps$^2$/km, $\gamma=$ "+str(gamma))
        
        plt.figure(8)
        FA=abs(np.fft.fft(A))**2
        plt.plot(Omega, FA,label=round(z*increments/h,2))
        plt.legend(title=r"Distance $L_D$ travelled")
        plt.ylabel(r"$|\tilde[A](z,T)|^2$, W")
        plt.xlabel(r"$\Omega$, THz")
        plt.xlim(-10, 10) 
        plt.title(r"Frequency Domain, $\beta_2$ =" +str(beta2)+ r"ps$^2$/km, $\gamma=$ "+str(gamma)) 

    A=split_step(A,h,gamma,beta2)
    z+=h
    

print("distance travelled:", round(1000*(z-h),4),"m")

#%% Different N value
N=3  #define soliton order
P_0=1       
T_0=1
beta2=-20
gamma=N**2*abs(beta2)/(P_0*T_0**2)   #define gamma based on N and beta_2

L_NL=1/(gamma*P_0)
L_D=T_0**2/abs(beta2)
print("Characteristic lengths:",L_D, L_NL)  #print characteristic lengths

A_0=np.sqrt(P_0)/(np.cosh((T/T_0)))         #unpropagated pusle given by sech shape

increments=0.05     #fraction of N_NL per h distance
h=increments*L_NL   #ensuring h is less than 5% of the smallest Lenghts scale

z_s=L_D*np.pi/2     #soltion period

A=A_0
z=0

repeats=400
h=z_s/repeats  #step distance, solition period in 400 increments


emptyA=np.zeros((repeats+1,len(T)))    #empty matrix to save temporal envelope at each step of loop
emptyF=np.zeros((repeats+1,len(T)))    # as above for frequency spectrum

for n in range (repeats+1):
    FA=abs(np.fft.fft(A))**2          #freq of pulse 
    emptyA[n,:]+=abs(A)**2            # add current temporal envelope to array
    emptyF[n,:]+=(FA)                 # as above for frequency
    A=split_step(A,h,gamma,beta2)

    z+=h
    
#plt.plot(Omega, emptyF[30,:])   
h_array=np.arange(0,z-h,h)/(np.pi/2)    #array of propagation distance

fig = plt.figure(9,figsize=(8,6))       #3d plot set up for temporal pulse
ax = plt.subplot(111, projection='3d')
ax.set_xlabel('T, ps')
ax.set_ylabel(r'$L_D/L_s$')
ax.set_zlabel(r"$|A(z,T)|^2$")
ax.set_xlim3d(-10, 10)
ax.set_title("Temporal domain for N="+str(N)+" Soliton")
ax.grid(False)

Omega_crop=np.concatenate((Omega[32698:32768],Omega[0:64]), axis=0) #take only useful frequencies, where peak is
emptyF_crop=np.concatenate((emptyF[:,32698:32768],emptyF[:,0:64]), axis=1)  #adjust for above change

X,Y =np.meshgrid(Omega_crop,h_array/L_D)
plot=ax.plot_surface(X=X,Y=Y,Z=emptyF_crop, cmap="jet")


fig = plt.figure(10,figsize=(8,6))    #3d plot set up for temporal pulse, same as above for frequency
ax = plt.subplot(111, projection='3d')
ax.set_xlabel(r"$\Omega$, THz")
ax.set_ylabel(r'$L_D/L_S$')
ax.set_zlabel(r"$|\tilde[A](z,T)|^2$")
ax.set_xlim3d(-5, 5)
ax.set_title("Frequency domain for N="+str(N)+" Soliton")
ax.grid(False)

T_crop=T[12288:20480]                      #crop for useful part of temporal window
emptyA_crop=emptyA[:,12288:20480]          # adjust for above

X,Y =np.meshgrid(T_crop,h_array/L_D)

plot=ax.plot_surface(X=X,Y=Y,Z=emptyA_crop, cmap="jet")
#%% Instability

P_0=1       
T_0=1
beta2=-20
gamma= 10

window=100*T_0
points=2**15
dt=window/points
T=np.arange(-window/2,window/2,dt)

Omega=2*np.pi*np.fft.fftfreq(len(T),dt) 

L_NL=1/(gamma*P_0)
L_D=T_0**2/abs(beta2)

Freq_shift=np.sqrt(-2*gamma*P_0/beta2)       #calculate modulation frequency, separation between combs
print("Characteristic lengths:",L_D, L_NL)

print("Omega=",Freq_shift)

noise=np.random.normal(1,0.1,len(T))        #create random noise signal to go over the top

#noise=Freq_shift                           # toggle between random noise and adding freq we know to be the modulation freq

#A_0=np.sqrt(P_0)*np.ones(len(T))+0.01*(np.exp(-j*Freq_shift*T)+np.exp(1j*Freq_shift*T))
A_0=np.sqrt(P_0)*np.ones(len(T))+0.01*(np.exp(-1j*noise*T)+np.exp(1j*noise*T))


increments=1/200   #factor deciding how many L_d are in an h step distance 
h=increments*L_D   #ensuring h is less than 5% of the smallest Lenghts scale

A=A_0     #reset pulse and distance travelled
z=0

repeats=int(11*L_D/h)         #number of loops
plot_numbers=[0, repeats]     #select which repeats to plot

for n in range (repeats+1):

    if n in plot_numbers:
        plt.figure(11)
        plt.plot(T, abs(A)**2, label=round(z*increments/(h),2))
        #plt.plot(T, (A**2*np.exp(-j*2*T)).real, color=colour[n])
        plt.legend(title=r"Distance $\L_{D}$ travelled")
        plt.ylabel(r"$|A(z,T)|^2, W$")
        plt.xlabel("T, ps")
        #plt.ylim(0,10)
        #plt.yscale('log')

        #plt.ylim(0,5)
        plt.title(r"Temporal Domain, with small signal")
        
        plt.figure(12)
        FA=abs(np.fft.fft(A))**2
        plt.plot(Omega, FA,label=round(z*increments/(h),2))
        plt.legend(title=r"Distance $L_{D}$ travelled")
        plt.ylabel(r"$|\tilde[A](z,T)|^2$, W")
        plt.xlabel(r"$\Omega$, THz")
        plt.xlim(-3, 3) 
        #plt.yscale('log')
        #plt.scal("log")
        plt.title(r"Frequency Domain, with small noise") 

    A=split_step(A,h,gamma,beta2)
    z+=h
    
print("distance travelled:", round(1000*(z-h),4),"m")



