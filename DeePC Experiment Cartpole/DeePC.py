import numpy as np
import numpy.matlib as mat
import cvxpy as cv
from scipy import sparse

class DeePC_Controller:

    def __init__(self, Yp, Yf, Up, Uf):

        '''
        Yp, Yf, Up, Uf: Hankel Matrices
        '''

        self.Yp = Yp
        self.Yf = Yf
        self.Up = Up
        self.Uf = Uf

    def OneNormDeePC(self, Q, R, lamda_g, lamda_s, ulims):

        '''
        Sets up problem for one norm regularised DeePC
        Q: array of costs
        R: array of costs
        lamda_g: factor for regularisation on g
        lamda_s: cost on slack
        ulims: limits on u
        '''

        # Properly Partitioned Hankel Matrix
        H = np.vstack((self.Up, self.Yp, self.Uf, self.Yf))
        # print(H.shape)
        # H = sparse.csc_matrix(M)

        # Cost Matrices
        R = sparse.diags(R)
        Q = sparse.diags(Q)

        # g vector
        g = cv.Variable((H.shape[1], 1))
        # print(g.shape)

        # slack vectors
        s = cv.Variable((self.Yp.shape[0], 1))

        # Initial conditions
        y_ini = cv.Parameter((self.Yp.shape[0], 1))
        u_ini = cv.Parameter((self.Up.shape[0], 1))

        # Predicted trajectory
        yf = cv.Variable((self.Yf.shape[0], 1))
        uf = cv.Variable((self.Uf.shape[0], 1))

        ref = cv.Parameter((self.Yf.shape[0], 1))

        obj = cv.quad_form(yf-ref, Q) + cv.quad_form(uf, R) + lamda_g*cv.norm1(g) + lamda_s*cv.norm1(s)

        constraints = [cv.vstack([u_ini, y_ini + s, uf, yf]) == H @ g]

        ulow = np.ones((uf.shape[0], 1))*ulims[0]
        uhigh = np.ones((uf.shape[0], 1))*ulims[1]
        # slow = np.ones((s.shape[0], 1))*lamda_s
        # shigh = -np.ones((s.shape[0], 1))*lamda_s

        constraints += [ulow <= uf, uf <= uhigh]
        #constraints += [slow <= s, s <= shigh]

        prob = cv.Problem(cv.Minimize(obj), constraints)

        return prob, y_ini, u_ini, ref, g, uf
    
    def OneNormDeePC_1(self, Q, R, lamda_g, eps_s, ulims, nx, Tini):

        '''
        Sets up problem for one norm regularised DeePC
        Q: array of costs
        R: array of costs
        lamda_g: factor for regularisation on g
        eps_s: slack limits
        ulims: limits on u
        '''

        # Properly Partitioned Hankel Matrix
        H = np.vstack((self.Up, self.Yp, self.Uf, self.Yf))

        # Cost Matrices
        R = sparse.diags(R)
        Q = sparse.diags(Q)

        # g vector
        g = cv.Variable((H.shape[1], 1))

        # slack vectors
        s = cv.Variable((self.Yp.shape[0], 1))
        # s_low = np.reshape((np.zeros((nx, Tini)) + eps_s).T, (-1, 1))
        # s_high = np.reshape((np.zeros((nx, Tini)) - eps_s).T, (-1, 1))

        # Initial conditions
        y_ini = cv.Parameter((self.Yp.shape[0], 1))
        u_ini = cv.Parameter((self.Up.shape[0], 1))

        # Predicted trajectory
        yf = cv.Variable((self.Yf.shape[0], 1))
        uf = cv.Variable((self.Uf.shape[0], 1))

        ref = cv.Parameter((self.Yf.shape[0], 1))

        obj = cv.quad_form(yf-ref, Q) + cv.quad_form(uf, R) + lamda_g*cv.norm1(g)

        constraints = [cv.vstack([u_ini, y_ini + s, uf, yf]) == H @ g]

        ulow = np.ones((uf.shape[0], 1))*ulims[0]
        uhigh = np.ones((uf.shape[0], 1))*ulims[1]

        constraints += [ulow <= uf, uf <= uhigh]
        
        constraints += [cv.sum(g) == 1]

        prob = cv.Problem(cv.Minimize(obj), constraints)

        return prob, y_ini, u_ini, ref, g, uf

