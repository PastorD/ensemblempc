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
t_max = 1.5                                                  # End time (sec)
t_eval = np.linspace(0, t_max, int(t_max/dt))               # Simulation time points


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
mean_sys = LinearSystemDynamics(A, B_mean)

print(f"Run Testing Ensemble Experiment")
N_sampling = 100
print(f"Main parameters: Nb:{Nb}, N_sampling:{N_sampling}, N_t:{N_steps}")
x_raw, x_ensemble = [], []
x_th, u_th  = [], []
impacts_ensemble, impacts_raw = [], []
B_hist = []
sigmaB = 0.2
for j in range(N_sampling):

    # Design robust MPC with current ensemble of Bs and execute experiment:
    #B_sample = B_mean + sigmaB @ np.random.randn(Ns,Nu)
    #B_hist.append(B_sample[1,0])

    mass_sample = 1/(1/mass+sigmaB*np.random.randn()) # perturb B_2=1/mass
    print(f"Test {j}: B:{mass_sample}")

    B_hist.append( [1/mass_sample] )
    system = LinearOneDimDrone(mass_sample, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)

    controller_ensemble = RobustMpcDense(mean_sys, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref,ensemble=B_ensemble, D=Dmatrix)          
    x_tmp, u_tmp = system.simulate(z_0, controller_ensemble, t_eval) 
    x_ensemble.append(x_tmp.T) # x_raw [N_sampling][Ns,Nt]_NumpyArray
    impacts_ensemble.append(system.impacts)
    
    controller_raw = RobustMpcDense(mean_sys, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref,ensemble=None, D=Dmatrix)          
    x_tmp, u_tmp = system.simulate(z_0, controller_raw, t_eval) 
    x_raw.append(x_tmp.T) # x_ensemble [N_sampling][Nt][Ns,]_NumpyArray
    impacts_raw.append(system.impacts)

    
#! Plot results
f21 = plt.figure(figsize=(12,6))
gs2 = gridspec.GridSpec(2,2, figure=f21)
# plt.subplot(2,2,1)
# plt.hist(B_hist)
# plt.xlabel("B(2,1)")
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.tight_layout()
# plt.grid()
colors = pl.cm.cool([(B[0]-min(B_hist)[0])/(max(B_hist)[0] - min(B_hist)[0]) for B in B_hist])

plt.subplot(2,2,1)
for x,impacts,colorB in zip(x_raw,impacts_raw,colors):
    plt.plot(t_eval,x[0,:],color=colorB) 
    [plt.scatter(impact[0],ground_altitude,color='b',s=40) for impact in impacts]
plt.plot( [1,t_max], [xmin[0],xmin[0]], '--r', lw=2, label='Ground')
plt.plot( [1,t_max], [ref[0,0],ref[0,0]], '--r', lw=1, label='Reference')
plt.xlabel("Time(s)")
plt.ylabel("Position(m)")
plt.title("Only considering the mean B")
plt.ylim(ground_altitude-0.1,ground_altitude+ 0.2)
plt.xlim(1,t_max)
plt.legend()
plt.grid()

plt.subplot(2,2,3)
for x,impacts,colorB in zip(x_raw,impacts_raw,colors):
    plt.plot(t_eval,x[1,:],color=colorB) 
    #[plt.scatter(impact[0],0,color='b',s=40) for impact in impacts]
plt.plot( [1,t_max], [0,0], '--r', lw=2, label='Zero Speed')
plt.xlabel("Time(s)")
plt.ylabel("Velocity(m/s)")
plt.ylim(-3.,2.)
plt.xlim(1,t_max)
plt.grid()


plt.subplot(2,2,2)
for x,impacts,colorB in zip(x_ensemble,impacts_ensemble,colors):
    plt.plot(t_eval,x[0,:],color=colorB) 
    [plt.scatter(impact[0],ground_altitude,color='b',s=40) for impact in impacts]
plt.plot( [1,t_max], [xmin[0],xmin[0]], '--r', lw=2, label='Ground')
plt.plot( [1,t_max], [ref[0,0],ref[0,0]], '--r', lw=1, label='Reference')
plt.xlabel("Time(s)")
plt.ylabel("Position(m)")
plt.title("Imposing constraints for all dynamics in the B ensemble")
plt.ylim(ground_altitude-0.1,ground_altitude+ 0.2)
plt.xlim(1,t_max)
plt.legend()
plt.grid()

plt.subplot(2,2,4)
for x,impacts,colorB in zip(x_ensemble,impacts_ensemble,colors):
    plt.plot(t_eval,x[1,:],color=colorB) 
    #[plt.scatter(impact[0],0,color='b',s=40) for impact in impacts]
plt.plot( [1,t_max], [0,0], '--r', lw=2, label='Zero Speed')
plt.xlabel("Time(s)")
plt.ylabel("Velocity(m/s)")
plt.ylim(-3.,2.)
plt.xlim(1,t_max)
plt.grid()

#plt.show()

f21.savefig('core/examples/results/test_B_both_noB.pdf', format='pdf', dpi=2400)

sp.io.savemat('./core/examples/1d_drone_Btesting.mat', {'x_ensemble': x_ensemble, 
                                                    'xmin':xmin, 
                                                    'x_raw':x_raw,
                                                    't_eval':t_eval, 
                                                    'B_hist':B_hist,
                                                    't_max':t_max,
                                                    'ref':ref})
