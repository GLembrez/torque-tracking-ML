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
delta       = Fsc/(r+D)
alpha       = D*delta + Fc
beta        = Fs*delta

## Joit friction computation ##

def friction(x) :
    if np.abs(x) <= Z*Fs:
        return x/Z
    else :
        a = D*Z**2 + Z
        bp = -(x+delta+2*D*Z*x+alpha*Z)
        bl = -(-x+delta-2*D*Z*x+alpha*Z)
        cp = D*x**2 + alpha*x + beta
        cl = D*x**2 - alpha*x + beta
        yp = (-bp - np.sqrt(bp**2-4*a*cp))/(2*a)
        yl = (bl + np.sqrt(bl**2-4*a*cl))/(2*a)

        if x > Z*Fs:
            return yp
        
        if x < -Z*Fs:
            return yl


simulation_on = True
e = 0
q = 0
states = []
qc = 3
Kp = 0.5
time = 0
while simulation_on :
    
    time += t
    q_new = q + Kp*t*(qc-q)
    q_dot = (q_new - q)/t
    q = q_new
    
    u = q_dot
    v_star = u + Z*K*e
    f = friction(v_star)
    # apply friction force to the system
    e = Z*(B*e+f*t)
    if time > 60 :
        simulation_on = False
    states.append(f)

fig = plt.figure()
plt.plot(states)
plt.show()
