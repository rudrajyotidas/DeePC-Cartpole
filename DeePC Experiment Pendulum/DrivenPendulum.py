import numpy as np
from scipy.integrate import RK45

class DrivenPendulum:

    def __init__(self, m, l, b=0):

        '''
        Initialise a simple driven pendulum object with parameters
        '''

        self.m = m
        self.l = l
        self.b = b
        self.g = 9.8

    def dynamics(self, X, u):

        '''
        X: Column vector of full state (theta and theta_dot)
        u: Control input (torque)

        Returns X_dot (theta_dot and theta_ddot)
        '''

        theta = X[0,0]
        theta_dot = X[1,0]

        theta_ddot = (u + self.b*theta_dot)/(self.m*(self.l**2)) - np.sin(theta)*self.g/self.l

        X_dot = np.array([theta_dot, theta_ddot])

        return X_dot[:, np.newaxis]
    
    def modelRK(self, t, X_U):

        '''
        For ease to interface with RK45, takes inputs in the form of row vector and returns a row vector
        Concatenated X and U together
        Takes time as input but not uses it because dynamics does not depend on time
        '''

        X = X_U[:2]
        u = X_U[2]

        X = X[:, np.newaxis]
        
        X_dot = self.dynamics(X, u)
        u_dot = 0

        return np.squeeze(np.vstack((X_dot, u_dot)))

    def TopDynamics(self):

        '''
        Returns A and B about unstable equilibrium point
        '''

        A = np.array([[0, 1],
                     [self.g/self.l, self.b/(self.m*(self.l**2))]])
        
        B = np.array([[0],
                      [1/(self.m*self.l*self.l)]])
        
        return A, B
        

class DrivenPendulumSimulator:

    def __init__(self, pend_obj:DrivenPendulum, Ts=0.05):
        
        '''
        Initialise with a Quadcopter object and a sampling time Ts
        Ts must be greater than 0.05
        '''

        self.pend_obj = pend_obj
        self.Ts = Ts

    def step(self, X, u):

        '''
        Takes row/column vectors X(t) and a number u(t) as input
        Returns row vector X(t+Ts)
        '''
        X0_u0 = np.concatenate((np.squeeze(X), np.array([u])))
        #X0_u0 = np.concatenate((np.squeeze(X), u))
        pend_sol = RK45(fun=self.pend_obj.modelRK, t0=0, y0=X0_u0, t_bound=self.Ts, max_step=0.01, rtol=1e-5)

        status = 'running'

        while status=='running':

            pend_sol.step()
            status = pend_sol.status

        return (pend_sol.y[:2])

