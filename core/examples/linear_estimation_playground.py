#%%
from ..learning import InverseKalmanFilter

import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp


def simulate(A,Bmean,Emean,sigmaB,sigmaE,T=100,dt=.1):
    """
    Simulate linear dynamics system
    
    Arguments:
        A {numpy array [n,n]} -- A in xdot = Ax+Bu+E
        Bmean {numpy array [n,m]} -- 
        Emean {numpy array [n,]} -- E
        sigmaB {float>0} -- standard deviation for B
        sigmaE {float>0} -- [description]
    
    Keyword Arguments:
        T {int>0} -- number of timesteps (default: {100})
        dt {float>0} -- timestep (default: {.1})
    
    Returns:
        [type] -- [description]
    """

    Ns = Bmean.shape[0]
    Nu = Bmean.shape[1]
    
    X = np.zeros((Ns,T))
    u = np.random.normal(size=(Nu,T))
    Bcollect = []
    for j in range(1,T):
        # Simulate dynamics
        B = Bmean + sigmaB @ np.random.randn(Ns,Nu)
        E = Emean + sigmaE @ np.random.randn(Ns)
        f = lambda t,x: A@x + B@u[:,j] + E
        X[:,j] = solve_ivp(f, [0, dt], X[:,j-1]).y[:, -1]
        Bcollect.append(B)
        
    t = np.linspace(0,dt*T,T)
    return X, u, t, Bcollect

A_mean = np.array([[0., 1.], [0., 0.]])
B_mean = np.array([[0.],[1]])
Ns = B_mean.shape[0]
E_mean = np.random.normal(Ns)
Nu = B_mean.shape[1]
sigmaB_traj = np.diag([0,0.2])
sigmaB_timestep = np.array([0.,0.])
sigmaE_traj = np.diag([0.0,0.])
sigmaE_timestep = np.diag([0.0,0.0])


dt = 0.01
Ntraj = 2
X, U, B = [],[], []
for i in range(Ntraj):
    B_traj = B_mean + sigmaB_traj @ np.random.randn(Ns,Nu)
    E_traj = E_mean + sigmaE_traj @ np.random.randn(Ns)
    print(f"Trajectory {i}, B[1]: {B_traj[1,0]}")
    X_temp, U_temp, t_temp, B_temp = simulate(A_mean,B_traj,E_traj,sigmaB_timestep,sigmaE_timestep,dt=dt)
    X.append(X_temp)
    U.append(U_temp)
    B.append(B_temp)

#%%

print(len(X))
eta_0 = 0.1
Nen = 5
bspread = 0.6
B_0 = np.zeros((Ns,Nu,Nen))
for i in range(Nen):
    B_0[:,:,i] = B_mean+np.random.randn(Ns,Nu)*bspread
    #B_0 = [B_mean + np.random.randn(Ns,Nu)*bspread for i in range(Ne)]
    
plt.hist([B[j][i][1,0] for i in range(99) for j in range(Ntraj)],bins=50)
plt.show()
eki = InverseKalmanFilter(A_mean,B_mean,E_mean,eta_0,B_0,dt=dt,nk=10)
eki.fit(X, X_dot=None,U=U)
cov_B = eki.eki.cov_theta
B_reco = np.mean(eki.B_ensemble_flat,axis=1)
print(f"B:{B_mean[1]} vs recovered:{B_reco[1]}")
print(f"sigma B_time:{sigmaB_timestep[1]}, sigma B_traj:{sigmaB_traj[1]}  vs recovered:{cov_B}")


    
#%%
#! ============================== PLOT RAW  =================================
case_name = "test"

plt.figure()
[plt.plot(t_temp, x1[0,:], linewidth=1,label="pos") for x1 in X]
plt.xlabel("Time (s)")
plt.ylabel("X")
plt.grid()
plt.legend()
plt.show()
plt.savefig(case_name+f"_all_states.pdf", format='pdf', dpi=1200,bbox_inches='tight')

