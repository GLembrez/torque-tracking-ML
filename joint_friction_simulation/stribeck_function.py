import numpy as np
from matplotlib import pyplot as plt
from random import random as rand







def friction(x,K,B,Fs,Fc,vs,D,t) :

    Z           = 1/(K*t+B)
    Fsc         = Fs-Fc
    r           = Fsc/vs - D
    gamma       = D
    delta       = vs
    alpha       = D*delta + Fc
    beta        = Fs*delta

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

def Stribeck(x,K,B,Fs,Fc,vs,D,t):

    Z           = 1/(K*t+B)
    Fsc         = Fs-Fc
    r           = Fsc/vs - D
    gamma       = D
    delta       = vs
    alpha       = D*delta + Fc
    beta        = Fs*delta


    if x<0:
        return -(gamma*x**2-alpha*x+beta)/(-x+delta)
    else:
        return (gamma*x**2+alpha*x+beta)/(x+delta)


x = np.linspace(-0.4,0.4,1000)
## Hyperparameters ##


for i in range(10) :
    K = 5000
    B = 50  
    Fs = 2.5  
    Fc = 0.2      
    vs = 0.04      
    D = 4.5
    t = 0.001  
    y = [friction(i,K,B,Fs,Fc,vs,D,t) for i in x]
    y_Stribeck = [Stribeck(i,K,B,Fs,Fc,vs,D,t) for i in x]
    fig_1 = plt.figure()
    ax1 = fig_1.add_subplot(121)
    ax1.plot(x,y_Stribeck,color = 'lightsalmon',label='Stribeck friction function',linewidth=2)
    ax1.plot(x,y,color = 'teal', label='Implicit transform',linewidth=2)
    K = 1000 + rand()*9000  
    B = 10 + rand()*90    
    Fs = 1 + rand()*5
    Fc = rand()*Fs       
    vs = 0.01 + rand()*0.1  
    D = rand()*10
    t = 0.001  
    y = [friction(i,K,B,Fs,Fc,vs,D,t) for i in x]
    y_Stribeck = [Stribeck(i,K,B,Fs,Fc,vs,D,t) for i in x]
    ax2 = fig_1.add_subplot(122)
    ax2.plot(x,y_Stribeck,color = 'lightsalmon',label='Stribeck friction function',linewidth=2)
    ax2.plot(x,y,color = 'teal', label='Implicit transform',linewidth=2)
    # plt.xlabel(r'$v$ / $v^*$ [rad/s]',fontsize=18)
    # plt.ylabel(r'f [Nm]',fontsize=18)
    # plt.legend(fontsize=18)
    # plt.grid(linewidth=0.5)
    plt.show()

# figK = plt.figure()
# D = np.linspace(0,20,20)
# for D_test in D :
#     y_Stribeck = [friction(i,K,B,Fs,Fc,vs,D_test,t) for i in x]
#     plt.plot(x,y_Stribeck, color = 'teal', linewidth=0.5)
# plt.xlabel(r'$v$ [rad/s]',fontsize=18)
# plt.ylabel(r'f [Nm]',fontsize=18)
# plt.grid(linewidth=0.5)
# plt.show()