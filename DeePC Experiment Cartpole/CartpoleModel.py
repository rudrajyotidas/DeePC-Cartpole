import numpy as np
from scipy.integrate import RK45

class Cartpole:

    def __init__(self, parameters):

        self.mp = parameters[0]
        self.mc = parameters[1]
        self.l = parameters[2]

        self.g = parameters[3]

    def ManipulatorEqns(self, state):

        x = state[0, 0]
        theta = state[1, 0]
        x_dot = state[2, 0]
        theta_dot = state[3, 0]

        M = np.array([[self.mc + self.mc, self.mp*self.l*np.cos(theta)],
                      [self.mp*self.l*np.cos(theta), self.mp*(self.l**2)]])
        
        C = np.array([[0, -self.mp*self.l*theta_dot*np.sin(theta)],
                      [0, 0]])
        
        G = np.array([[0],
                      [-self.mp*self.g*self.l*np.sin(theta)]])
        
        B = np.array([[1],
                      [0]])
        
        return M, B, C, G
    
    def dynamics(self, X, u):

        '''
        X: Column vector of full state (4 elements)
        U: Column vector of control inputs (1 element)

        Returns X_dot
        '''

        q = X[:2, :]
        q_dot = X[2:, :]

        M, B, C, G = self.ManipulatorEqns(X)

        q_ddot = np.linalg.inv(M)@(B*u + G - C@q_dot)

        return np.vstack((q_dot, q_ddot))
    
    def modelRK(self, t, X_U):

        '''
        For ease to interface with RK45, takes inputs in the form of row vector and returns a row vector
        Concatenated X and U together
        Takes time as input but not uses it because dynamics does not depend on time
        '''

        X = X_U[:4]
        U = X_U[4]

        X = X[:, np.newaxis]

        X_dot = self.dynamics(X, U)
        U_dot = 0

        return np.squeeze(np.vstack((X_dot, U_dot)))
    
    def LinearisedDynamics(self):

        dG_dq = np.array([[0, 0],
                          [0, self.mp*self.g*self.l]])
        
        M, B, _, _ = self.ManipulatorEqns(np.array([[0],
                                                    [np.pi],
                                                    [0],
                                                    [0]]))
        
        A = np.block([[np.zeros((2,2)), np.eye(2)],
                      [np.linalg.inv(M)@(dG_dq), np.zeros((2,2))]])
        
        B = np.block([[np.zeros((2,1))],
                      [np.linalg.inv(M)@B]])
        
        return A, B
    
class CartpoleSimulator:

    def __init__(self, pend_obj:Cartpole, Ts=0.01):

        '''
        Initialise with a Cartpole object and sampling time
        Ts must be greater than 0.01
        '''

        self.pend_obj = pend_obj
        self.Ts = Ts

    def step(self, X, u):

        '''
        Takes row/column vectors X(t) and a number u(t) as input
        Returns row vector X(t+Ts)
        '''

        X0_u0 = np.concatenate((np.squeeze(X), np.array([u])))

        pend_sol = RK45(fun=self.pend_obj.modelRK, t0=0, y0=X0_u0, t_bound=self.Ts, max_step=0.001, rtol=1e-5)

        status = 'running'

        while status=='running':

            pend_sol.step()
            status = pend_sol.status

        return (pend_sol.y[:4])