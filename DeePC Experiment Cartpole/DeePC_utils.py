import numpy as np

def createHankel(traj, L, T):

    '''
    Creates Hankel matrix from a trajectory
    traj: n by T matrix containing a time evolution trajectory
    L: Length of each sub-trajectory
    T: Total length of trajectory (number of timesteps)
    '''
    n = traj[:, 0].shape[0]
    Hankel = np.zeros((n*L, T-L+1))

    for i in range(T-L+1):

        Hankel[:, i] = np.reshape(traj[:, i:i+L].T, (-1))

    return Hankel

def mergedHankel(tupleofHankels):

    return np.hstack(tupleofHankels)

def splitHankel(H, Tini, L, n):

    '''
    H: Hankel matrix
    n: number of elements in each vector
    Tini: part for initial trajectory
    L: Length of each subtrajectory in H
    '''

    Hp = H[:Tini*n, :]
    Hf = H[Tini*n:, :]

    return Hp, Hf

def TimeShift(y_prev, y):

    '''
    y_prev: y for the last Tini timesteps
    y: measured y
    Returns new time shifted trajectory with y appended
    '''

    y_new = np.hstack((y_prev[:, 1:], y))

    return y_new

def ReshapeTraj(traj):

    '''
    Reshapes trajectory for ease
    '''

    return np.reshape(traj.T, (-1,1))

def giveReference(traj, timestep, window):

    ref = ReshapeTraj(traj[:, timestep:timestep+window])

    return ref