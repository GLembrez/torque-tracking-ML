import numpy as np
from matplotlib import pyplot as plt

## Hyperparameters ##

K           = 5000      # stiffness coefficient 
B           = 50        # damping coefficient
t           = 0.001     # time discretization
Fs          = 2.5       # static friction 
Fc          = 0.2       # kinetic friction
Fsc         = Fs-Fc
vs          = 0.04      # break-away velocity
D           = 4.5       # viscosity coefficient
Z           = 1/(K*t+B)

## Simplified Stribeck parameters ##

r           = Fsc/vs - D
gamma       = D
delta       = vs
alpha       = D*delta + Fc
beta        = Fs*delta

## Joit friction computation ##

def friction(x) :
    if np.abs(x) <= Z*Fs:
        return x/Z
    else :
        a = D*Z**2 + Z
        bp = -(x+delta+2*D*Z*x+alpha*Z)
        bl = (-x+delta-2*D*Z*x+alpha*Z)
        cp = D*x**2 + alpha*x + beta
        cl = D*x**2 - alpha*x + beta
        yp = (-bp - np.sqrt(bp**2-4*a*cp))/(2*a)
        yl = (-bl + np.sqrt(bl**2-4*a*cl))/(2*a)

        if x > Z*Fs:
            return yp
        
        if x < -Z*Fs:
            return yl

def Stribeck(x):

    if x<0:
        return -(gamma*x**2-alpha*x+beta)/(-x+delta)
    else:
        return (gamma*x**2+alpha*x+beta)/(x+delta)


x = np.linspace(-0.4,0.4,1000)
y = [friction(i) for i in x]
y_Stribeck = [Stribeck(i) for i in x]

fig = plt.figure()
plt.plot(x,y_Stribeck,color = 'lightsalmon',label='Stribeck friction function',linewidth=2)
plt.plot(x,y,color = 'teal', label='Implicit transform',linewidth=2)
plt.xlabel(r'$v$ / $v^*$ [rad/s]',fontsize=18)
plt.ylabel(r'f [Nm]',fontsize=18)
plt.legend(fontsize=18)
plt.grid(linewidth=0.5)
plt.show()