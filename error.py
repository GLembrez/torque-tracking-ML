import numpy as np

class Friction():
    def __init__(self,K,B,Fs_lim,Fc_lim,dFs_lim,dFc_lim,vs_lim,D_lim):
        self.e      = np.zeros(7)               # velocity error due to friction
        self.f      = np.zeros(7)               # friction torque
        self.v      = np.zeros(7)               # hypothetical frictionless velocity
        self.fixed  = np.zeros(7)               # fixed point of frictin function
        self.Fs     = np.zeros(7)               # Force required to escape stiction
        self.Fc     = np.zeros(7)               # Force required to continue motion
        self.dFs    = np.zeros(7)               # Coupling coefficient for effect of actuator torque
        self.dFc    = np.zeros(7)               # Coupling coefficient for effect of actuator torque
        self.Fcgrav = np.zeros(7)               # Coupling coefficient for gravity torque
        self.Fsgrav = np.zeros(7)               # Coupling coefficient for gravity torque 
        self.vs     = np.zeros(7)               # velocity when escaping stiction
        self.D      = np.zeros(7)               # coefficient for viscous friction
        self.dt     = 0.001                     # simulation time step

        self.K = K                              # coefficient of elasiticity 
        self.B = B                              # damping of the material
        self.Z = 1/(self.dt*self.K + self.B)    # Impedance of elasto-plastic material
        self.Fs_lim = Fs_lim                    #
        self.Fc_lim = Fc_lim                    #
        self.dFs_lim = dFs_lim                  # Domain randomization :
        self.dFc_lim = dFc_lim                  # Bounds for sampling the friction model coefficients
        self.vs_lim = vs_lim                    #
        self.D_lim = D_lim                      #

        self.model = None
        self.data = None
        self.controller = None

        self.t = 0
        self.T = 10000                          # Iteration at which to invoke domain randomization

        self.randomize()

    def register(self,model,data,controller) :
        self.model = model
        self.data = data 
        self.controller = controller 


    def randomize(self):
        """
        Computes domain randomization on the analytical friction model
        The coefficients of the Stribeck model are sampled from uniform distribution
        """
        self.Fs = np.random.uniform(self.Fs_lim[0],self.Fs_lim[1],7)
        self.Fc = np.random.uniform(self.Fc_lim[0],0.8*self.Fs,7)
        self.vs = np.random.uniform(self.vs_lim[0],self.vs_lim[1],7)
        self.D = np.random.uniform(self.D_lim[0],self.D_lim[1],7) 
        self.dFs = np.random.uniform(self.dFs_lim[0],self.dFs_lim[1],7)
        self.dFc = np.random.uniform(self.dFc_lim[0],self.dFc_lim[1],7)
        self.t = 0

    def compute(self,tau,e):
        """
        Computes friction and velocity error at the joints

        f - vector of friction torques
        e - vector of velocity errors
        """
        f = np.zeros(7)
        for i in range(7):
            f[i],e[i] = self.compute_joint(tau[i],e[i],i)
        return f,e
    

    def compute_target(self,tau):
        """
        Computes ideal friction at the joints

        target - vector of friction torques
        """
        target = np.zeros(7)
        for i in range(7):
            target[i] = self.compute_target_joint(tau[i],i)
        return target
    

    def compute_joint(self,tau,e,i):
        """
        computes analytical friction at joint i using Stribeck model

        tau     - actuator torque at the joint
        c       - gravity coriolis torque
        v_rot   - rotation velocity of the joint
        e       - velocity error due to friction
        v       - hypothetical velocity if the joint was frictionless
        dtau    - torque due to friction
        """
        # update relevant quantities from simulation
        c = self.data.qfrc_bias[i]
        v_rot = self.data.qvel[i]
        self.v[i] =  v_rot + self.Z * self.K * e 
        Fs_coupled = self.Fs[i] + np.abs(self.dFs[i]*(tau-c))
        Fc_coupled = self.Fc[i] + np.abs(self.dFc[i]*(tau-c))
        alpha = self.D[i] * self.vs[i] + Fc_coupled
        beta = Fs_coupled * self.vs[i]
        dtau = 0

        # use simpler Stribeck model (rational)
        if np.abs(self.v[i]) <= self.Z * Fs_coupled:
            dtau =  self.v[i]/self.Z
        else:
            a = self.D[i]*self.Z**2 + self.Z
            bp = -(self.v[i]+self.vs[i]+2*self.D[i]*self.Z*self.v[i]+alpha*self.Z)
            bl = (-self.v[i]+self.vs[i]-2*self.D[i]*self.Z*self.v[i]+alpha*self.Z)
            cp = self.D[i]*self.v[i]**2 + alpha*self.v[i] + beta
            cl = self.D[i]*self.v[i]**2 - alpha*self.v[i] + beta
            if self.v[i] > self.Z*Fs_coupled:
                dtau = (-bp - np.sqrt(bp**2-4*a*cp))/(2*a) 
            if self.v[i] < -self.Z*Fs_coupled:
                dtau = (-bl + np.sqrt(bl**2-4*a*cl))/(2*a)
        
        # update velocity error
        e = self.Z * (self.B * e + dtau * self.dt)
        return dtau,e
        
    def compute_target_joint(self,tau,i):
        """
        Computes ideal friction in joint i
        Ideal friction is the friction which would occur if the joint i was moving at the desired speed
        
        dtau  - friction torque from Stribeck model
        alpha - desired velocity 
        tau   - actuator torque at the joint
        c     - gravity coriolis torque
        """
        alpha = self.controller.alpha_d[i]
        c = self.data.qfrc_bias[i]
        E = np.exp(- np.abs(alpha/self.vs[i]))
        S = np.sign(alpha)
        dtau = ((self.Fs[i] + self.dFs[i]*np.abs(tau - c) ) - \
            (self.Fc[i] + self.dFc[i]*np.abs(tau - c) )) * E * S + \
            (self.Fc[i]+ self.dFc[i]*np.abs(tau - c))*S + self.D[i]*alpha
        return dtau
    
    def compute_target_fixed_point(self,tau_array):
        """
        Computes the exact fixed point of the ideal friction torque using the analytical expression of the Stribeck model

        fixed - the fixed point
        """
        for i in range(7):
            tau = tau_array[i]
            alpha = self.controller.alpha_d[i]
            c = self.data.qfrc_bias[i]
            E = np.exp(- np.abs(alpha/self.vs[i]))
            S = np.sign(alpha)
            if tau>c:
                Fs_tilde = self.Fs[i] + self.dFs[i]*(tau-c)
                Fc_tilde = self.Fc[i] + self.dFc[i]*(tau-c)
                self.fixed[i] = (self.D[i]*alpha + (Fs_tilde-Fc_tilde)*E*S + Fc_tilde*S)/\
                    (1 - (self.dFs[i]-self.dFc[i])*E*S - self.dFc[i]*S)
            else:
                Fs_tilde = self.Fs[i] - self.dFs[i]*(tau-c)
                Fc_tilde = self.Fc[i] - self.dFc[i]*(tau-c)
                self.fixed[i] = (self.D[i]*alpha + (Fs_tilde-Fc_tilde)*E*S + Fc_tilde*S)/\
                    (1 - (self.dFc[i]-self.dFs[i])*E*S + self.dFc[i]*S)

    
    def update(self,e,f):
        """
        stores friction and velocity error before computing fixed point
        """
        self.e = e
        self.f = f
        self.t += 5

    def evaluate(self,dtau,i):
        """
        evaluation of the f(x)-x function where f is the real friction function
        """
        tau = self.controller.cmd_tau[i]
        f,_ = self.compute_joint(tau+dtau,self.e[i],i)
        return f-dtau


    def find_fixed_point(self):
        """
        computes friction fixed point using Newton Raphson algorithm

        F       - the function to be minimized: such that F(x) = f(x) - x
        epsilon - increment of torque to compute first order derivative
        tol     - threshold on the norm of F for completion
        """
        for i in range(7):
            n_iter = 0
            tol = 1e-2
            epsilon = 1e-6
            grad = 0
            dtau = self.f[i]
            F = self.evaluate(dtau,i)
            while np.abs(F)>tol and n_iter < 100:   
            # Newton Raphson loop
                dtau += epsilon
                F_new,_ = self.evaluate(dtau,i)
                grad =  (F_new - F) / epsilon  -1
                dtau -=  F/grad
                F = self.evaluate(dtau,i)
                n_iter += 1
            self.fixed[i] = dtau