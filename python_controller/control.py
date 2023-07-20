import numpy as np

class Controller():
    def __init__(self):
        self.q_lims = np.array([[-np.pi,np.pi],
                                [-np.pi/2,np.pi/2],
                                [-np.pi,np.pi],
                                [-np.pi/2,np.pi/2],
                                [-np.pi,np.pi],
                                [-2,2],
                                [-np.pi,np.pi]])
        self.q_f = np.zeros(7)
        self.q_s = np.zeros(7)
        self.q_d = np.zeros(7)
        self.alpha_d = np.zeros(7)
        self.cmd_tau = np.zeros(7)
        self.K_p = 1e2 * np.array([1,1,1,1,0.5,0.5,0.5])
        self.K_d = 1e1 * np.array([1,1,1,1,0.5,0.5,0.5])
        self.t = 0
        self.dt = 0.001
        self.T = 2500  

        self.model = None
        self.data = None
        self.randomize()

    def register_model(self,model,data) :
        self.model = model  
        self.data = data
        
    def randomize(self):
        self.q_s = self.q_f
        self.q_f = np.random.uniform(self.q_lims[:,0], self.q_lims[:,1], size=7)
        self.t = 0

    def input(self):
        q = self.data.qpos
        alpha = self.data.qvel
        c = self.data.qfrc_bias
        q_v = (self.q_f + self.q_s)/2
        T_v = self.T/2
        a12 = (-3*(self.q_f - q_v) + 9*(q_v - self.q_s))/(4*T_v**2)
        a13 = (3*(self.q_f - q_v) - 5*(q_v - self.q_s))/(4*T_v**3)
        a21 = (3*(self.q_f - self.q_s))/(4*T_v)
        a22 = (3*(self.q_f - q_v) - 3*(q_v - self.q_s))/(2*T_v**2)
        a23 = (-5*(self.q_f - q_v) + 3*(q_v - self.q_s))/(4*T_v**3)
        if self.t<self.T/2:
            self.q_d = self.t**3 * a13 + self.t**2 * a12 + self.q_s
            self.alpha_d = 1/self.dt * (3*self.t**2 * a13 + 2*self.t * a12 )
        else:
            self.q_d = (self.t-self.T/2)**3 * a23 + (self.t-self.T/2)**2 * a22 + (self.t-self.T/2) * a21 + q_v
            self.alpha_d = 1/self.dt *(3*(self.t-self.T/2)**2 * a23 + 2*(self.t-self.T/2) * a22 + a21)

        self.cmd_tau  = self.K_p * (self.q_d-q) + self.K_d * (self.alpha_d - alpha) + c
        self.t += 1