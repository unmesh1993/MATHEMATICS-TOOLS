import numpy as np
import os
import sympy as sym
from scipy.optimize import *

def objective(x,a0,a1,a2,a3,a4):
    eq = a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4
    return eq

def curve_fitting(e12,e2,a0,a1,a2,a3,a4):
    start=[a0,a1,a2,a3,a4]

    print ('a0,a1,a2,a3,a4')
#    print ('Input:   ', a0,a1,a2,a3,a4)

    popt, pcov = curve_fit (objective,e12,e2,p0=start)
    a0,a1,a2,a3,a4 = popt

    print ('output:  ', a0,a1,a2,a3,a4)
#    print (pcov)
    return a0,a1,a2,a3,a4

res2='data.dat'
f=open(res2,'r')
a=f.readlines()
f.close

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

a0,a1,a2,a3
a0,a1,a2=curve_fitting(e12,e2,a0,a1,a2)
ee2=objective(e12,a0,a1,a2)

# Compute chi square
r = (e2 - ee2)
chisq = np.sum((r)**2)
print("chisq =", chisq/(len(e2)-3))

print ('minima[pos]', e12[np.argmin(ee2)])

#####################################################################################################################################################################
x = sym.symbols("x")
ffo=a0 + a1*x + a2*x**2
gradient = sym.derive_by_array(ffo, (x))
#hessian = sym.Matrix(1, 1, sym.derive_by_array(gradient, (x)))
stationary_points = sym.solve(gradient, (x))
print ('stationary points   :',stationary_points)