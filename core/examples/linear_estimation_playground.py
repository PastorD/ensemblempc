from ..learning import InverseKalmanFilter

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp


def simulate(A,B,E,T=100,dt=.1):

    Ns = B.shape[0]
    Nu = B.shape[1]
    
    X = np.zeros((Ns,T))
    u = np.random.normal(size=(Nu,T))
    for j in range(1,T):
        # Simulate dynamics
        f = lambda t,x: A@x + B@u[:,j] + E
        X[:,j] = solve_ivp(f, [0, dt], X[:,j-1]).y[:, -1]
        
    t = np.linspace(0,dt*T,T)
    return X,t

A_mean = np.array([[0., 1.], [0., 0.]])
B_mean = np.array([[0.],[1]])
Ns = B_mean.shape[0]
E_mean = np.random.normal(Ns)
Nu = B_mean.shape[1]



X,t = simulate(A_mean,B_mean,E_mean)

#! ============================== PLOT RAW  =================================
case_name = "test"

plt.figure
plt.plot(t, X.T, linewidth=1,label='True State')
plt.xlabel("Time (s)")
plt.ylabel("X")
plt.grid()
plt.legend()
plt.show()
plt.savefig(case_name+f"_all_states.pdf", format='pdf', dpi=1200,bbox_inches='tight')

