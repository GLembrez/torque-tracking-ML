import numpy as np

class Friction():
    def __init__(self,K_lim,B_lim,Fs_lim,Fc_lim,dFs_lim,dFc_lim,vs_lim,D_lim):
        self.e      = np.zeros(7)       # velocity error due to friction
        self.f      = np.zeros(7)       # friction torque
        self.v      = np.zeros(7)       # hypothetical frictionless velocity
        self.fixed  = np.zeros(7)       # fixed point of frictin function
        self.Z      = np.zeros(7)       # Impedance of elasto-plastic material
        self.K      = np.zeros(7)       # coefficient of elasiticity 
        self.B      = np.zeros(7)       # damping of the material
        self.Fs     = np.zeros(7)       # Force required to escape stiction
        self.Fc     = np.zeros(7)       # Force required to continue motion
        self.dFs    = np.zeros(7)       # Coupling coefficient for effect of actuator torque
        self.dFc    = np.zeros(7)       # Coupling coefficient for effect of actuator torque
        self.vs     = np.zeros(7)       # velocity when escaping stiction
        self.D      = np.zeros(7)       # coefficient for viscous friction
        self.dt     = 0.001             # simulation time step

        self.K_lim = K_lim
        self.B_lim = B_lim
        self.Fs_lim = Fs_lim
        self.Fc_lim = Fc_lim
        self.dFs_lim = dFs_lim
        self.dFc_lim = dFc_lim
        self.vs_lim = vs_lim
        self.D_lim = D_lim

        self.model = None
        self.data = None
        self.controller = None

        self.randomize()

    def register(self,model,data,controller) :
        self.model = model
        self.data = data 
        self.controller = controller 


    def randomize(self):
        # Computes domain randomization on the analytical friction model
        self.K = np.random.uniform(self.K_lim[0],self.K_lim[1],7)
        self.B = np.random.uniform(self.B_lim[0],self.B_lim[1],7)
        self.Fs = np.random.uniform(self.Fs_lim[0],self.Fs_lim[1],7)
        self.Fc = np.random.uniform(self.Fc_lim[0],self.Fs,7)
        self.vs = np.random.uniform(self.vs_lim[0],self.vs_lim[1],7)
        self.D = np.random.uniform(self.D_lim[0],self.D_lim[1],7) 
        self.dFs = np.random.uniform(self.dFs_lim[0],self.dFs_lim[1],7)
        self.dFc = np.random.uniform(self.dFc_lim[0],self.dFc_lim[1],7)

    def compute(self,tau,e):
        # computes friction using Stribeck model
        # tau is the actuator torque at the joints
        # v_rot is the rotation velocity of the joint
        # e is the velocity error due to friction
        v_rot = self.data.qvel
        f = np.zeros(7)
        self.Z = 1/(self.dt*self.K + self.B)
        self.v = v_rot + self.Z * self.K * e
        Fs_coupled = self.Fs + np.abs(self.dFs*tau)
        Fc_coupled = self.Fc + np.abs(self.dFc*tau)
        alpha = self.D * self.vs + Fc_coupled
        beta = Fs_coupled * self.vs
        for i in range(7) :
            if np.abs(self.v[i]) <= self.Z[i] * Fc_coupled[i]:
                f[i] =  self.v[i]/self.Z[i]
            else:
                a = self.D[i]*self.Z[i]**2 + self.Z[i]
                bp = -(self.v[i]+self.vs[i]+2*self.D[i]*self.Z[i]*self.v[i]+alpha[i]*self.Z[i])
                bl = (-self.v[i]+self.vs[i]-2*self.D[i]*self.Z[i]*self.v[i]+alpha[i]*self.Z[i])
                cp = self.D[i]*self.v[i]**2 + alpha[i]*self.v[i] + beta[i]
                cl = self.D[i]*self.v[i]**2 - alpha[i]*self.v[i] + beta[i]
                if self.v[i] > self.Z[i]*Fs_coupled[i]:
                    f[i] = (-bp - np.sqrt(bp**2-4*a*cp))/(2*a) 
                if self.v[i] < -self.Z[i]*Fs_coupled[i]:
                    f[i] = (-bl + np.sqrt(bl**2-4*a*cl))/(2*a)
        e = self.Z * (self.B * e + f * self.dt)
        return f,e
    
    def update(self,e,f):
        # stores friction and velocity error before computing fixed point
        self.e = e
        self.f = f

    def evaluate(self,dtau):
        tau = self.controller.cmd_tau
        v_rot = self.data.qvel
        f,_ = self.compute(tau+dtau,self.e)
        return f-dtau
    
    def find_fixed_point(self):
        # computes friction fixed point using Newton Raphson algorithm
        tau = self.controller.cmd_tau
        v_rot = self.data.qvel
        n_iter = 0
        tol = 1e-2
        epsilon = 1e-3
        lr = 2
        J = np.zeros((7,7))
        dtau = self.f.copy()
        F = self.evaluate(dtau)
        while np.linalg.norm(F)>tol and n_iter < 10 :   
        # Newton Raphson loop
            for i in range(7) :
                dx = np.zeros(7)
                dx[i] = epsilon
                F_new = self.evaluate(dtau+dx)
                J[:,i] = (F_new - F) / epsilon 
                J[i,i] -= 1 
            dtau -= lr * np.linalg.inv(J).dot(F)
            F = self.evaluate(dtau)
            n_iter += 1
        self.fixed = dtau