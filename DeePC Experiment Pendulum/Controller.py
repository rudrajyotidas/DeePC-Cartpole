import numpy as np
from scipy import linalg as la

class Controller:

    def __init__(self, A, B, m, g, l):

        '''
        Initialise using A and B matrices post linearisation
        '''

        self.A = A
        self.B = B
        self.m = m
        self.g = g
        self.l = l

    def K_LQR(self, Q, R):

        '''
        Returns LQR Gain
        u = K@x
        '''

        P = la.solve_discrete_are(self.A, self.B, Q, R)
        K = -np.linalg.inv(self.B.T@P@self.B + R)@self.B.T@P@self.A

        return K
    
    def SwingUp(self, theta, theta_dot, k=0.1):

        E_curr = 0.5*self.m*(pow(self.l,2))*(pow(theta_dot,2)) - self.m*self.g*self.l*np.cos(theta)
        E_d = self.m*self.g*self.l

        u = -k*theta_dot*(E_curr - E_d)

        return u