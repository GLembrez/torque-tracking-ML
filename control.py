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
        self.q = np.zeros(7)
        self.alpha = np.zeros(7)
        self.alpha_d = np.zeros(7)
        self.inertia_corr = np.zeros(7)
        self.cmd_tau = np.zeros(7)
        self.t = np.zeros(7)
        self.T = np.zeros(7)  
        self.K_p = 1e2 * np.array([1,1,1,1,0.5,0.5,0.5])
        self.K_d = 1e1 * np.array([1,1,1,1,0.5,0.5,0.5])
        self.dt = 0.001

        self.model = None
        self.data = None
        

    def register_model(self,model,data) :
        self.model = model  
        self.data = data
        for i in range(7):
            self.randomize(i)
        
    def randomize(self,i):
        """
        - Selects joint i
        - Samples a new target posture in the joint space using uniform distribution
        - Sample a random duration of the new trajectory
        """
        self.q_s[i] = self.q_f[i]
        self.q_f[i] = np.random.uniform(self.q_lims[i,0], self.q_lims[i,1], size=1)
        self.t[i] = 0
        self.T[i] = int(np.random.uniform(2500,10000,size=1))


    def input(self):
        """
        Computes the desired trajectories and associated desired torques from the state of the robot and the desired posture
        The position of the joints follows a third order polyomial
        The velocity of the joits follows a second order polynomial
        The motion is continuous

        q       - position of the joints
        q_s     - starting position
        q_f     - desired final position
        q_v     - position at half trajectory
        alpha   - velocity of the joints
        c       - gravity coriolis vector
        T       - vector of the durations of the trajectories
        T_v     - time at half trajectory
        t       - vector of the current time in each trajectories: t[i] in (0,T[i])
        K_p     - proportional gains
        K_d     - derivative gains
        cmd_tau - desired torque 
        """

        # Gather current state from simulation
        self.q = self.data.qpos.copy()
        self.alpha = self.data.qvel.copy()
        c = self.data.qfrc_bias
        q_v = (self.q_f + self.q_s)/2
        T_v = self.T/2

        # Compute polynomial trajectories to ensure smooth motion
        a12 = (-3*(self.q_f - q_v) + 9*(q_v - self.q_s))/(4*T_v**2)
        a13 = (3*(self.q_f - q_v) - 5*(q_v - self.q_s))/(4*T_v**3)
        a21 = (3*(self.q_f - self.q_s))/(4*T_v)
        a22 = (3*(self.q_f - q_v) - 3*(q_v - self.q_s))/(2*T_v**2)
        a23 = (-5*(self.q_f - q_v) + 3*(q_v - self.q_s))/(4*T_v**3)

        # update desired position and velocity
        for i in range(7):
            if self.t[i]<self.T[i]/2:
                self.q_d[i] = self.t[i]**3 * a13[i] + self.t[i]**2 * a12[i] + self.q_s[i]
                self.alpha_d[i] = 1/self.dt * (3*self.t[i]**2 * a13[i] + 2*self.t[i] * a12[i] )
            else:
                self.q_d[i] = (self.t[i]-self.T[i]/2)**3 * a23[i] + (self.t[i]-self.T[i]/2)**2 * a22[i] + (self.t[i]-self.T[i]/2) * a21[i] + q_v[i]
                self.alpha_d[i] = 1/self.dt *(3*(self.t[i]-self.T[i]/2)**2 * a23[i] + 2*(self.t[i]-self.T[i]/2) * a22[i] + a21[i])

        # update desired torque using PD correction
        self.cmd_tau  = self.K_p * (self.q_d-self.q) + self.K_d * (self.alpha_d - self.alpha)  + c 
        self.t += 1