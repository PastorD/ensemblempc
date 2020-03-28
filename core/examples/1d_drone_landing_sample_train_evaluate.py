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
np.random.seed(127)
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
ref_training = np.array([[ground_altitude+0.1 for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])


#! Filter Parameters:
eta = 0.6**2 # measurement covariance
Nb = 3 # number of ensemble
nk = 5 # number of steps for multi-step prediction
# B_ensemble = np.zeros((Ns,Nu,Nb))
# for i in range(Nb):
#     B_ensemble[:,:,i] = B_mean+np.array([[0.],[np.random.uniform(-0.5,0.5)]])

E = np.array([0,-gravity*mass])


#B_ensemble_list = [B_mean-np.array([[0.],[0.5]]), B_mean, B_mean+np.array([[0.],[0.5]])]
print(f"Main parameters: Nb:{Nb}, N_ep:{N_ep}, N_t:{N_steps}")


#! ===============================================   Gather Data   ================================================

print(f"Run sample train evaluate Experiment")
N_sampling = 3
x_data, u_data = [], []
B_hist = []
sigmaB =0.3
for j in range(N_sampling):
    # Change trajectory every time
    ref_training = np.array([[ground_altitude+0.1+2*np.random.rand() for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])
    Q_training = np.array([[1e4*(1+3*np.random.rand()), 0.], [0., 1.]])

    mass_sample = 1/(1/mass+sigmaB*np.random.randn()) # perturb B_2=1/mass

    B_hist.append( [1/mass_sample] )
    system = LinearOneDimDrone(mass_sample, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)


    print(f"Test {j}: mass:{mass_sample}")
    lin_dyn = LinearSystemDynamics(A, B_mean)
    
    controller_training = RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref_training,ensemble=None, D=Dmatrix)          
    x_tmp, u_tmp = system.simulate(z_0, controller_training, t_eval) 
    x_data.append(x_tmp.T) # x_data [N_sampling][Ns,Nt]_NumpyArray
    u_data.append(u_tmp.T) # u_data [N_sampling][Ns,Nt]_NumpyArray

    
#! Plot results
f21 = plt.figure(figsize=(18,9))
gs2 = gridspec.GridSpec(3,2, figure=f21)

colors = pl.cm.cool([(B[0]-min(B_hist)[0])/(max(B_hist)[0] - min(B_hist)[0]) for B in B_hist])
plt.subplot(3,1,1)
[plt.plot(t_eval,x[0,:],color=colorB) for x,colorB in zip(x_data,colors)]
plt.plot( [1,t_max], [xmin[0],xmin[0]], '--r', lw=2, label='Ground')
#plt.plot( [1,t_max], [ref[0,0],ref[0,0]], '--r', lw=1, label='Reference')
plt.xlabel("Time(s)")
plt.ylabel("Position(m)")
plt.title("Only considering the mean B")
#plt.ylim(ground_altitude-0.1,ground_altitude+ 0.2)
#plt.xlim(1,t_max)
plt.legend()
plt.grid()

plt.subplot(3,1,2)
[plt.plot(t_eval,x[1,:],color=colorB) for x,colorB in zip(x_data,colors)]
plt.plot( [1,t_max], [0,0], '--r', lw=2, label='Zero Speed')
plt.xlabel("Time(s)")
plt.ylabel("Velocity(m/s)")
#plt.ylim(-3.,2.)
#plt.xlim(1,t_max)
plt.grid()

plt.subplot(3,1,3)
[plt.plot(t_eval[:-1],u[0,:],color=colorB) for u,colorB in zip(u_data,colors)]
plt.plot( [1,t_max], [0,0], '--r', lw=2, label='Zero Speed')
plt.xlabel("Time(s)")
plt.ylabel("U")
plt.grid()


plt.show()
f21.savefig('core/examples/results/data.pdf', format='pdf', dpi=2400)


sp.io.savemat('./core/examples/1d_drone_Btesting.mat', {'u_data': u_data, 
                                                    'xmin':xmin, 
                                                    'x_data': x_data,
                                                    't_eval':t_eval, 
                                                    'B_hist':B_hist,
                                                    't_max':t_max})


#! ===============================================   Fit Ensemble   ================================================

B_ensemble_0 = np.stack([B_mean-np.array([[0.],[0.6]]), B_mean, B_mean+np.array([[0.],[0.6]])],axis=2)
inverse_kalman_filter = InverseKalmanFilter(A,B_mean, E, eta, B_ensemble_0, dt, nk )
inverse_kalman_filter.fit(x_data, u_data) 
B_ensemble = inverse_kalman_filter.B_ensemble

print(B_ensemble)
print(f"B mean sigma: {inverse_kalman_filter.eki.cov_theta}")
B_reco = np.mean(inverse_kalman_filter.B_ensemble_flat,axis=1)
print(f"B used mean:{B_mean[1]} vs recovered:{B_reco[1]}")
#! ===============================================   Evaluate   ================================================

print(f"Run sample train evaluate Experiment")
N_testing = 3
x_data, u_data = [], []
B_hist = []
sigmaB =0.3
for j in range(N_sampling):
    # Change trajectory every time
    ref_training = np.array([[ground_altitude+0.1 for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])
    Q_training = np.array([[1e4*(1+3*np.random.rand()), 0.], [0., 1.]])

    mass_sample = 1/(1/mass+sigmaB*np.random.randn()) # perturb B_2=1/mass

    B_hist.append( [1/mass_sample] )
    system = LinearOneDimDrone(mass_sample, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)


    print(f"Test {j}: mass:{mass_sample}")
    lin_dyn = LinearSystemDynamics(A, B_mean)
    
    controller_training = RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref_training,ensemble=B_ensemble, D=Dmatrix)          
    x_tmp, u_tmp = system.simulate(z_0, controller_training, t_eval) 
    x_data.append(x_tmp.T) # x_data [N_sampling][Ns,Nt]_NumpyArray
    u_data.append(u_tmp.T) # u_data [N_sampling][Ns,Nt]_NumpyArray

#! Plot results
f21 = plt.figure(figsize=(18,9))
gs2 = gridspec.GridSpec(3,2, figure=f21)

colors = pl.cm.cool([(B[0]-min(B_hist)[0])/(max(B_hist)[0] - min(B_hist)[0]) for B in B_hist])
plt.subplot(3,1,1)
[plt.plot(t_eval,x[0,:],color=colorB) for x,colorB in zip(x_data,colors)]
plt.plot( [1,t_max], [xmin[0],xmin[0]], '--r', lw=2, label='Ground')
#plt.plot( [1,t_max], [ref[0,0],ref[0,0]], '--r', lw=1, label='Reference')
plt.xlabel("Time(s)")
plt.ylabel("Position(m)")
plt.title("Only considering the mean B")
plt.ylim(ground_altitude-0.1,ground_altitude+ 0.2)
plt.xlim(1,t_max)
plt.legend()
plt.grid()

plt.subplot(3,1,2)
[plt.plot(t_eval,x[1,:],color=colorB) for x,colorB in zip(x_data,colors)]
plt.plot( [1,t_max], [0,0], '--r', lw=2, label='Zero Speed')
plt.xlabel("Time(s)")
plt.ylabel("Velocity(m/s)")
plt.ylim(-3.,2.)
plt.xlim(1,t_max)
plt.grid()

f21.savefig('core/examples/results/evaluate.pdf', format='pdf', dpi=2400)