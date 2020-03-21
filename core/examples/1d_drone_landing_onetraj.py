from ..systems import OneDimDrone, LinearOneDimDrone
from ..controllers import RobustMpcDense, MPCController, OpenLoopController
from ..dynamics import SystemDynamics, LinearSystemDynamics
from ..learning import InverseKalmanFilter

from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

#%%
print("Starting 1D Drone Landing Simulation..")
#! ===============================================   SET PARAMETERS    ================================================
# Define system parameters of the drone:
mass = 1                                                    # Drone mass (kg)
rotor_rad = 0.08                                            # Rotor radius (m)
drag_coeff = 0.5                                            # Drag coefficient
air_dens = 1.25                                             # Air density (kg/m^3)
area = 0.04                                                 # Drone surface area in xy-plane (m^2)
gravity = 9.81                                              # Gravity (m/s^2)
T_hover = mass*gravity                                      # Hover thrust (N)
ground_altitude = 0.2                                       # Altitude corresponding to drone landed (m)
#system = OneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)
system = LinearOneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)

# Define initial linearized model and ensemble of Bs (linearized around hover):
A = np.array([[0., 1.], [0., 0.]])
B_mean = np.array([[0.],[1/mass]])
Ns = B_mean.shape[0]
Nu = B_mean.shape[1]

# Define simulation parameters:
z_0 = np.array([4., 0.])                                    # Initial position
dt = 1e-2                                                   # Time step length
t_max = 2.5                                                  # End time (sec)
t_eval = np.linspace(0, t_max, int(t_max/dt))               # Simulation time points
N_ep = 6                                                   # Number of episodes

# Model predictive controller parameters:
Q = np.array([[1e4, 0.], [0., 1.]])
QN = Q
R = np.array([[1.]])
Dmatrix = sp.sparse.diags([50000,30000])
N_steps = int(t_max/dt)-1
umin = np.array([-T_hover])
umax = np.array([30.-T_hover])
xmin=np.array([ground_altitude, -5.])
xmax=np.array([10., 5.])
ref = np.array([[ground_altitude+0.1 for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])


#! Filter Parameters:
eta = 0.6**2 # measurement covariance
Nb = 3 # number of ensemble
nk = 5 # number of steps for multi-step prediction
# B_ensemble = np.zeros((Ns,Nu,Nb))
# for i in range(Nb):
#     B_ensemble[:,:,i] = B_mean+np.array([[0.],[np.random.uniform(-0.5,0.5)]])

E = np.array([0,-gravity*mass])
B_ensemble = np.stack([B_mean-np.array([[0.],[0.6]]), B_mean, B_mean+np.array([[0.],[0.6]])],axis=2)


#B_ensemble_list = [B_mean-np.array([[0.],[0.5]]), B_mean, B_mean+np.array([[0.],[0.5]])]
true_sys = LinearSystemDynamics(A, B_mean)
print(f"Main parameters: Nb:{Nb}, N_ep:{N_ep}, N_t:{N_steps}")

#! == Run limited MPC Controller ============

print(f"Run test Hard EnMPC")
lin_dyn_mean = LinearSystemDynamics(A, B_mean)
ctrl_tmp_mean = RobustMpcDense(lin_dyn_mean, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ensemble)

ctrl_tmp_mean.eval(z_0, 0)
u_mean = ctrl_tmp_mean.get_control_prediction()
z_mean = ctrl_tmp_mean.get_state_prediction()
z_b = ctrl_tmp_mean.get_ensemble_state_prediction()

t_z = np.linspace(0,ctrl_tmp_mean.N*dt,ctrl_tmp_mean.N)
f,axarr=plt.subplots(3, sharex=True)
f.subplots_adjust(hspace=.1)

plt.subplot(3,1,1,ylabel='Position')
plt.plot(t_z, z_mean[0,:], linewidth=3, label=f'B Mean {B_mean}')
for i in range(Nb):
    plt.plot(t_z, z_b[i][0,:], linewidth=1, label=f'B {B_ensemble[:,:,i]}')
plt.plot(t_z, ground_altitude*np.ones(t_z.shape), linestyle="--", linewidth=1, label=f'Minimum z', color='gray')
plt.legend(loc='upper right')
plt.grid()

plt.subplot(3,1,2, ylabel='Velocity')
plt.plot(t_z, z_mean[1,:], linewidth=3, label=f'B Mean {B_mean}')
for i in range(Nb):
    plt.plot(t_z, z_b[i][1,:], linewidth=1, label=f'B {B_ensemble[:,:,i]}')
plt.plot(t_z, xmin[1]*np.ones(t_z.shape), linestyle="--", linewidth=1, color='gray')
plt.plot(t_z, xmax[1]*np.ones(t_z.shape), linestyle="--", linewidth=1, color='gray')
plt.legend(loc='upper right')
plt.grid()

plt.subplot(3,1,3, xlabel="Time(s)",ylabel='Control Input')
plt.plot(t_z,u_mean.T,label=f'U Optimizing Mean Dynamics')
plt.plot(t_z, umin*np.ones(t_z.shape), linestyle="--", linewidth=1, color='gray')
plt.plot(t_z, umax*np.ones(t_z.shape), linestyle="--", linewidth=1, color='gray')
plt.legend(loc='upper right')
plt.grid()
plt.show()
f.savefig('core/examples/results/hard_mpc.pdf', format='pdf', dpi=2400)