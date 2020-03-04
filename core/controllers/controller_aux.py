
from numpy.linalg import eigvals
import time

import numpy as np
import scipy as sp
import osqp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .controller import Controller
from ..learning.edmd import Edmd

# Auxiliary functions to support Controllers


def block_diag(M, n):
    """bd creates a sparse block diagonal matrix by repeating M n times

    Args:
        M (2d numpy array): matrix to be repeated
        n (float): number of times to repeat
    """
    return sp.sparse.block_diag([M for i in range(n)])


def build_boldAB(Ad, Bd, Nt):
    # it computes the matrices to obtain all future states given an initial state and a control action sequence
    # as in x = bold_a @ x_0 + bold_b @ u
    # x [Nt x Ns,]
    # bold_a [Nt x Ns, Ns]
    # bold_b [Nt x Ns, Nt Nu]
    # u [Nt x Nu]
    #
    # all flat matrices are done using row-major order ('C'), that is:
    # [1 2 3
    #  4 5 6] => [1 2 3 4 5 6]


    Ns = Bd.shape[0]
    Nu = Bd.shape[1]

    Bbd = block_diag(Bd, Nu).tocoo()

    #! GET a & b
    # Write B:
    diag_AkB = Bd
    data_list = Bbd.data
    row_list = Bbd.row
    col_list = Bbd.col
    for i in range(Nt):
        if i < Nt-1:
            AkB_bd_temp = block_diag(diag_AkB, Nt-i)
        else:
            AkB_bd_temp = diag_AkB.tocoo()
        data_list = np.hstack([data_list, AkB_bd_temp.data])
        row_list = np.hstack(
            [row_list, AkB_bd_temp.row+np.full((AkB_bd_temp.row.shape[0],), Ns*i)])
        col_list = np.hstack([col_list, AkB_bd_temp.col])

        diag_AkB = Ad.dot(diag_AkB)

    bold_B = sp.sparse.coo_matrix(
        (data_list, (row_list, col_list)), shape=(Nt*Ns, Nt*Nu))

    #! Build bold_A
    bold_A = Ad.copy()
    Ak = Ad.copy()
    for i in range(Nt-1):
        Ak = Ak.dot(Ad)
        bold_A = sp.sparse.vstack([bold_A, Ak])

    return bold_A, bold_B

def test_build_boldAB(Ad, Bd, Nt):

    Ns = Bd.shape[0]
    Nu = Bd.shape[1]

    bA, bB = build_boldAB(Ad, Bd, Nt)
    x0  = np.linspace(-5,40,Ns)
    x00 = np.linspace(-5,40,Ns)
    # Store data Init
    nsim = N
    xst = np.zeros((Ns,nsim))
    ust = np.zeros((Nu,nsim))

    # Simulate in closed loop

    for i in range(nsim):
        # Fake pd controller
        ctrl = np.zeros(Nu,) #np.random.rand(nu,)
        x0 = Ad.dot(x0) + Bd.dot(ctrl)

        # Store Data
        xst[:,i] = x0
        ust[:,i] = ctrl

    x_dense = np.reshape(bA @ x00 + bB @ (ust.flatten('F')),(Nt,Ns)).T

    plt.figure()
    plt.subplot(2,1,1)
    for i in range(Ns):
        plt.plot(range(nsim),xst[i,:],'d',label="sim "+str(i))
        plt.plot(range(nsim),x_dense[i,:],'d',label="ax+bu "+str(i))
    plt.xlabel('Time(s)')
    plt.grid()
    plt.legend()

    plt.subplot(2,1,2)
    for i in range(nu):
        plt.plot(range(nsim),ust[i,:],label=str(i))
    plt.xlabel('Time(s)')
    plt.grid()
    plt.legend()
    plt.savefig("AB_check.pdf",bbox_inches='tight',format='pdf', dpi=2400)
    plt.close()
