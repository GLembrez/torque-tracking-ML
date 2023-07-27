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
        self.Fcgrav = np.zeros(7)       # Coupling coefficient for gravity torque
        self.Fsgrav = np.zeros(7)       # Coupling coefficient for gravity torque 
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
        self.Z = 1/(self.dt*self.K + self.B)

    def compute(self,tau,e):
        f = np.zeros(7)
        for i in range(7):
            f[i],e[i],_ = self.compute_joint(tau[i],e[i],i)
        return f,e
    

    def compute_joint(self,tau,e,i):
        # computes friction using Stribeck model
        # tau is the actuator torque at the joints
        # v_rot is the rotation velocity of the joint
        # e is the velocity error due to friction
        # fixed_exists is a boolean 0 if fixed point does not exist
        fixed_exists = True
        c = self.data.qfrc_bias[i]
        v_rot = self.data.qvel[i]
        self.v[i] = v_rot + self.Z[i] * self.K[i] * e
        f = 0
        Fs_coupled = self.Fs[i] + np.abs(self.dFs[i]*(tau-c))
        Fc_coupled = self.Fc[i] + np.abs(self.dFc[i]*(tau-c))
        alpha = self.D[i] * self.vs[i] + Fc_coupled
        beta = Fs_coupled * self.vs[i]
        f = 0
        if np.abs(self.v[i]) <= self.Z[i] * Fc_coupled:
            f =  self.v[i]/self.Z[i]
            fixed_exists = False
        else:
            a = self.D[i]*self.Z[i]**2 + self.Z[i]
            bp = -(self.v[i]+self.vs[i]+2*self.D[i]*self.Z[i]*self.v[i]+alpha*self.Z[i])
            bl = (-self.v[i]+self.vs[i]-2*self.D[i]*self.Z[i]*self.v[i]+alpha*self.Z[i])
            cp = self.D[i]*self.v[i]**2 + alpha*self.v[i] + beta
            cl = self.D[i]*self.v[i]**2 - alpha*self.v[i] + beta
            if self.v[i] > self.Z[i]*Fs_coupled:
                f = (-bp - np.sqrt(bp**2-4*a*cp))/(2*a) 
            if self.v[i] < -self.Z[i]*Fs_coupled:
                f = (-bl + np.sqrt(bl**2-4*a*cl))/(2*a)
        e = self.Z[i] * (self.B[i] * e + f * self.dt)
        return f,e,fixed_exists
        
    
    def update(self,e,f):
        # stores friction and velocity error before computing fixed point
        self.e = e
        self.f = f

    def evaluate(self,dtau,i):
        tau = self.controller.cmd_tau[i]
        v_rot = self.data.qvel[i]
        f,_,exists = self.compute_joint(tau+dtau,self.e[i],i)
        return f-dtau,exists


    def find_fixed_point(self):
        # computes friction fixed point using Newton Raphson algorithm
        for i in range(7):
            n_iter = 0
            tol = 1e-2
            epsilon = 1e-6
            lr = 2
            grad = 0
            dtau = self.f[i]
            F,exists = self.evaluate(dtau,i)
            if exists:
                while np.abs(F)>tol and n_iter < 10 :   
                # Newton Raphson loop
                    F_new,_ = self.evaluate(dtau,i)
                    grad =  (F_new - F) / epsilon  -1
                    dtau -= lr * F/grad
                    F = F_new
                    n_iter += 1
            self.fixed[i] = dtau