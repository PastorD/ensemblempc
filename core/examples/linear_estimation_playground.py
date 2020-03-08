from ..learning import InverseKalmanFilter

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp


def simulate(A,Bmean,Emean,sigmaB,sigmaE,T=100,dt=.1):

    Ns = Bmean.shape[0]
    Nu = Bmean.shape[1]
    
    X = np.zeros((Ns,T))
    u = np.random.normal(size=(Nu,T))
    for j in range(1,T):
        # Simulate dynamics
        B = Bmean + sigmaB @ np.random.randn(Ns,Nu)
        E = Emean + sigmaE @ np.random.randn(Ns)
        f = lambda t,x: A@x + B@u[:,j] + E
        X[:,j] = solve_ivp(f, [0, dt], X[:,j-1]).y[:, -1]
        
    t = np.linspace(0,dt*T,T)
    return X, u, t

A_mean = np.array([[0., 1.], [0., 0.]])
B_mean = np.array([[0.],[1]])
Ns = B_mean.shape[0]
E_mean = np.random.normal(Ns)
Nu = B_mean.shape[1]
sigmaB = np.diag([0,0.2])
sigmaE = np.diag([0.0,0.0])


dt = 0.01
Ntraj = 20 
X, U = [],[]
for i in range(Ntraj):
    X_temp, U_temp, t_temp = simulate(A_mean,B_mean,E_mean,sigmaB,sigmaE,dt=dt)
    X.append(X_temp)
    U.append(U_temp)
    
print(len(X))
eta_0 = 0.1
Ne = 5
bspread = 0.6
B_0 = [B_mean + np.random.randn(Ns,Nu)*bspread for i in range(Ne)]
#eki = InverseKalmanFilter(A_mean,B_mean,E_mean,eta_0,B_0,dt=dt,nk=0)
#eki.fit(X, X_dot=None,U=U)
#cov_B = eki.get_Cov_theta()
#print(f"$\sigma_B:${sigmaB[1]} vs recovered:{cov_B[0]}")


    

#! ============================== PLOT RAW  =================================
case_name = "test"

plt.figure
[plt.plot(t_temp, x1, linewidth=1,label="pos") for x1 in X]
plt.xlabel("Time (s)")
plt.ylabel("X")
plt.grid()
plt.legend()
plt.show()
plt.savefig(case_name+f"_all_states.pdf", format='pdf', dpi=1200,bbox_inches='tight')

