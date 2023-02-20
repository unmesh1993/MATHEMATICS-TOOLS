# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 21:41:36 2018

@author: unmesh
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D # This import has side effects required for the kwarg projection='3d' in the call to fig.add_subplot
import matplotlib.pyplot as plt
import random
import sys
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import sympy as sym
from scipy.optimize import *







###################################################################################################
def objective(x,a0,a1,a2,a3,a4):
    eq = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4
    return eq

def objective1(x,a0,a1,a2):
    eq = a0 + a1*x + a2*x**2
    return eq


###################################################################################################
def curve_fitting(e12,e2,a0,a1,a2,a3,a4):
    start=[a0,a1,a2,a3,a4]

    print ('a0,a1,a2,a3,a4')
#    print ('Input:   ', a0,a1,a2,a3,a4)

    popt, pcov = curve_fit (objective,e12,e2,p0=start)
    a0,a1,a2,a3,a4 = popt

    print ('output:  ', a0,a1,a2,a3,a4)
#    print (pcov)
    return a0,a1,a2,a3,a4

def curve_fitting1(e12,e2,a0,a1,a2):
    start=[a0,a1,a2]

    print ('a0,a1,a2')
#    print ('Input:   ', a0,a1,a2)

    popt, pcov = curve_fit (objective1,e12,e2,p0=start)
    a0,a1,a2 = popt

    print ('output:  ', a0,a1,a2)
#    print (pcov)
    return a0,a1,a2


sec=['70K', '100K']
const=400
##################################################################

res2='data.dat'
f=open(res2,'r')
a=f.readlines()
f.close

###########BOMD
fig=plt.figure()
for i in range(0,2):
    e12=[]
    e2=[]
    e2s=[]
    for j in range(100):
        t=j+i*(100+1)
        x=a[t].split()
        if float(x[1]) < const:
           e12.append(x[0])
           e2.append(x[1])
           e2s.append(x[2])

    e12=np.array(e12,float)
    e2=np.array(e2,float)
    e2s=np.array(e2s,float)

    a0=1.0
    a1=1.0
    a2=1.0
    a3=1.0
    a4=1.0


    if i == 0 :

           a0,a1,a2,a3
           a0,a1,a2=curve_fitting1(e12,e2,a0,a1,a2)
           ee2=objective1(e12,a0,a1,a2)

           # Compute chi square
           r = (e2 - ee2)
           chisq = np.sum((r)**2)
           print("chisq =", chisq/(len(e2)-3))


#           params=a0,a1,a2
#           grid=(np.amin(e12),np.amax(e12),np.absolute(e12[0]-e12[1]))
#           xmin_global = brute(objective1,(grid,),args=params,full_output=True,finish=optimize.fmin )
#           print ('minima',xmin_global[0])

           print ('minima[pos]', e12[np.argmin(ee2)])

#####################################################################################################################################################################
           x = sym.symbols("x")
           ffo=a0 + a1*x + a2*x**2
           gradient = sym.derive_by_array(ffo, (x))
#           hessian = sym.Matrix(1, 1, sym.derive_by_array(gradient, (x)))
           stationary_points = sym.solve(gradient, (x))
           print ('stationary points   :',stationary_points)
#####################################################################################################################################################################



    else:
           a0,a1,a2,a3,a4=curve_fitting(e12,e2,a0,a1,a2,a3,a4)
           ee2=objective(e12,a0,a1,a2,a3,a4)

           # Compute chi square
           r = (e2 - ee2)
           chisq = np.sum((r)**2)
           print("chisq =", chisq/(len(e2)-5))




           mingy=np.amin(ee2)
           mingx=e12[np.argmin(ee2)]



           if i > 0:
              dog1=ee2[(e12 > 0.0)]
              dog2=e12[(e12 > 0.0)]
              minly=np.amin(dog1)
              minlx=dog2[np.argmin(dog1)]

              dog1=ee2[(e12 < minlx) & (e12 > mingx)]
              dog2=e12[(e12 < minlx) & (e12 > mingx)]
              barry=np.amax(dog1)
              barrx=dog2[np.argmax(dog1)]

              print ('pos[minima-g,maxima,minima-g]: ', mingx , barrx , minlx )
              print ('Act. E[forward,bakward,Asym. E (eV)]: ', barry-mingy,barry-minly,(barry-mingy-(barry-minly)))

#####################################################################################################################################################################
           x = sym.symbols("x")
           ffo=a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4
           gradient = sym.derive_by_array(ffo, (x))
#           hessian = sym.Matrix(1, 1, sym.derive_by_array(gradient, (x)))
           stationary_points = sym.solve(gradient, (x))
           print ('stationary points   :',stationary_points)
#####################################################################################################################################################################


   #           params=a0,a1,a2,a3,a4
   #           grid=(np.amin(e12),np.amax(e12),np.absolute(e12[0]-e12[1]))
   #           xmin_global = brute(objective,(grid,),args=params,full_output=True,finish=optimize.fmin )
   #           print ('minima-g',xmin_global[0])
   #           grid=(0,np.amax(e12),np.absolute(e12[0]-e12[1]))
   #           xmin_local = brute(objective,(grid,),args=params,full_output=True,finish=optimize.fmin )
   #           print ('minima-l',xmin_local[0])
   #
   #
   #           grid=(xmin_global[0],xmin_local[0],np.absolute(e12[0]-e12[1]))
   #           xmax_global = brute(lambda x,a0,a1,a2,a3,a4: -objective(x,a0,a1,a2,a3,a4),(grid,),args=params,full_output=True,finish=optimize.fmin )
   #           print ('maxima',xmax_global[0],-np.amin(xmax_global[3]))
   #           print (xmax_global)

    plt.plot(e12,ee2, lw=2 ,mfc='white')


    plt.scatter(e12,e2, s=20,label=sec[i])


    ###############################################################################

    plt.ylabel('Effective free energy (meV)',size=10)
    plt.xlabel(r'$\delta_o $ (Å)',size=10)
    plt.xticks(np.arange(-1.0,1.1,0.2),size=10)
    plt.yticks(size=10)
    plt.xlim(-0.9,0.9)
    plt.ylim(-10,250)
    plt.axvline(x=0.0, color = 'k',linewidth=1, linestyle = '--')
    plt.legend()
    print ('--------------------------------------------------------------------------------------------------------------------------------------')


plt.subplots_adjust(wspace=0.4,hspace=0.4)
fig.savefig('BOMD.pdf')
plt.savefig('BOMD.jpg', dpi=300, bbox_inches = 'tight',    pad_inches = 0.1)

####################################################################################################################################




###########PIGLET
fig=plt.figure()
for i in range(0,2):
    e12=[]
    e2=[]
    e2s=[]
    for j in range(100):
        t=j+(i+2)*(100+1)
        x=a[t].split()
        if float(x[1]) < const:
           e12.append(x[0])
           e2.append(x[1])
           e2s.append(x[2])

    e12=np.array(e12,float)
    e2=np.array(e2,float)
    e2s=np.array(e2s,float)

    a0=1.0
    a1=1.0
    a2=1.0
    a3=1.0
    a4=1.0

    a0,a1,a2,a3,a4=curve_fitting(e12,e2,a0,a1,a2,a3,a4)
    ee2=objective(e12,a0,a1,a2,a3,a4)


    # Compute chi square
    r = (e2 - ee2)
    chisq = np.sum((r)**2)
    print("chisq =", chisq/(len(e2)-5))

    mingy=np.amin(ee2)
    mingx=e12[np.argmin(ee2)]

    dog1=ee2[(e12 > 0.0)]
    dog2=e12[(e12 > 0.0)]
    minly=np.amin(dog1)
    minlx=dog2[np.argmin(dog1)]

    dog1=ee2[(e12 < minlx) & (e12 > mingx)]
    dog2=e12[(e12 < minlx) & (e12 > mingx)]
    barry=np.amax(dog1)
    barrx=dog2[np.argmax(dog1)]

    print ('pos[minima-g,maxima,minima-g]: ', mingx , barrx , minlx )
    print ('Act. E[forward,bakward,Asym. E (eV)]: ', barry-mingy,barry-minly,(barry-mingy-(barry-minly)))


#####################################################################################################################################################################
    x = sym.symbols("x")
    ffo=a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4
    gradient = sym.derive_by_array(ffo, (x))
#    hessian = sym.Matrix(1, 1, sym.derive_by_array(gradient, (x)))
    stationary_points = sym.solve(gradient, (x))
    print ('stationary points   :',stationary_points)
#####################################################################################################################################################################



#    grid=(np.amin(e12),np.amax(e12),np.absolute(e12[0]-e12[1]))
#    params=a0,a1,a2,a3,a4
#    xmin_global = brute(objective,(grid,),args=params,full_output=True,finish=optimize.fmin )
#    print ('minima-g',xmin_global[0])
#    grid=(0,np.amax(e12),np.absolute(e12[0]-e12[1]))
#    xmin_local = brute(objective,(grid,),args=params,full_output=True,finish=optimize.fmin )
#    print ('minima-l',xmin_local[0])
#
#    grid=(xmin_global[0],xmin_local[0],np.absolute(e12[0]-e12[1]))
#    xmax_global = brute(lambda x,a0,a1,a2,a3,a4: -objective(x,a0,a1,a2,a3,a4),(grid,),args=params,full_output=True,finish=optimize.fmin )
#    print ('maxima',xmax_global[0])


    plt.plot(e12,ee2, lw=2 ,mfc='white')
    plt.scatter(e12,e2, s=20,label=sec[i])


    ###############################################################################

    plt.ylabel('Effective free energy (meV)',size=10)
    plt.xlabel(r'$\delta_o $ (Å)',size=10)
    plt.xticks(np.arange(-1.0,1.1,0.2),size=10)
    plt.yticks(size=10)
    plt.xlim(-0.9,0.9)
    plt.ylim(-10,100)
    plt.axvline(x=0.0, color = 'k',linewidth=1, linestyle = '--')
    plt.legend()
    print ('--------------------------------------------------------------------------------------------------------------------------------------')

plt.subplots_adjust(wspace=0.4,hspace=0.4)
fig.savefig('PIGLET.pdf')
plt.savefig('PIGLET.jpg', dpi=300, bbox_inches = 'tight',    pad_inches = 0.1)

####################################################################################################################################