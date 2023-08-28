import numpy as np
from scipy import linalg as la

class Controller:

    def __init__(self, A, B, parameters):

        '''
        Initialise using A and B matrices post linearisation
        '''

        self.A = A
        self.B = B
        self.mp = parameters[0]
        self.mc = parameters[1]
        self.l = parameters[2]
        self.g = parameters[3]

    def K_LQR(self, Q, R):

        '''
        Returns LQR Gain
        u = K@x
        '''

        P = la.solve_discrete_are(self.A, self.B, Q, R)
        K = -np.linalg.inv(self.B.T@P@self.B + R)@self.B.T@P@self.A

        return K
    
    def SwingUp(self, kE, kP, kD, state):

        x = state[0, 0]
        theta = state[1, 0]
        x_dot = state[2, 0]
        theta_dot = state[3, 0]

        E_d = self.mp*self.g*self.l

        E_curr = 0.5*(self.mp + self.mc)*x_dot**2 + 0.5*self.mp*(self.l**2)*(theta_dot**2) \
                    + self.mp*x_dot*theta_dot*self.l*np.cos(theta) - self.mp*self.g*self.l*np.cos(theta)
        
        u = kE*theta_dot*np.cos(theta)*(E_curr - E_d) - kP*x - kD*x_dot

        return u