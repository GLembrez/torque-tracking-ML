import mc_control
import mc_rbdyn
import mc_rtc
import mc_tasks
import numpy as np

class McController(mc_control.MCPythonController):
    def __init__(self, rm, dt):
        self.qpsolver.addConstraintSet(self.dynamicsConstraint)
        self.qpsolver.addConstraintSet(self.contactConstraint)
        self.qpsolver.addTask(self.postureTask)

        self.targetPosition = mc_tasks.PostureTask(self.qpsolver,self.robot().robotindex(), 1,1)
        self.qpsolver.addTask(self.targetPosition)
        self.targetPosition.reset()

        self.pGain = np.array([300,300,300,300,150,150,150])
        self.pGain = np.array([9,9,9,9,1.5,1.5,1.5])
        self.q_lims = np.array([[-np.pi,np.pi],
                                [-np.pi/2,np.pi/2],
                                [-np.pi,np.pi],
                                [-np.pi/2,np.pi/2],
                                [-np.pi,np.pi],
                                [-2,2],
                                [-np.pi,np.pi]])
        self.q = np.zeros(7)
        self.alpha = np.zeros(7)
        self.c = np.zeros(7)
        self.q_f = np.zeros(7)
        self.q_s = np.zeros(7)
        self.q_d = np.zeros(7)
        self.alpha_d = np.zeros(7)
        self.cmd_tau = np.zeros(7)
        self.t = 0
        self.dt = 0.001
        self.T = 2500 

        self.logger().addLogEntry("cmd_tau", lambda: self.cmd_tau)

    def run_callback(self):
        if self.targetPosition.eval().norm() < 0.1 or self.t>10 :
            self.switch_target()
            self.t = 0
        forward_dynamics = mc_rbdyn.ForwardDynamics(self.robot().mb)
        forward_dynamics.computeC(self.realRobot().mb, self.realRobot().mbc())
        self.observe()
        return 
    
    def observe(self):
        forward_dynamics = mc_rbdyn.ForwardDynamics(self.robot().mb)
        forward_dynamics.computeC(self.realRobot().mb, self.realRobot().mbc())
        self.q = self.param_to_vector(self.realRobot().mbc.q)
        self.alpha = self.param_to_vector(self.realRobot().mbc.alpha)
        self.c = forward_dynamics.C
    
    def reset_callback(self):
        self.targetPosition.reset()
    
    def switch_target(self):
        pass

    def randomize(self):
        self.q_s = self.q_f
        self.q_f = np.random.uniform(self.q_lims[:,0], self.q_lims[:,1], size=7)
        self.t = 0

    def input(self):
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

        self.cmd_tau  = self.K_p * (self.q_d-self.q) + self.K_d * (self.alpha_d - self.alpha) + self.c
        self.t += 1

    def param_to_vector(self,param):
        x = np.zeros(7)
        for i in range(7):
            x[i] = param[i][0]
        return x

    @staticmethod
    def create(robot, dt):
        env = mc_rbdyn.get_robot_module("env", mc_rtc.MC_ENV_DESCRIPTION_PATH, "ground")
        return McController([robot,env], dt)

