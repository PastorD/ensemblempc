from ..systems import OneDimDrone, LinearOneDimDrone
from ..controllers import RobustMpcDense, MPCController, OpenLoopController
from ..dynamics import SystemDynamics, LinearSystemDynamics
from ..learning import InverseKalmanFilter, Monomials, Edmd

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from control import lqr

#%%
print("Starting 1D Drone Landing Simulation...")
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
system = OneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)
#system = LinearOneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)

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
N_ep = 10                                                   # Number of episodes

# Model predictive controller parameters:
Q = np.array([[1e4, 0.], [0., 1e1]])
QN = Q
R = np.array([[1e1]])
Dmatrix = sp.sparse.diags([50000,30000])
N_steps = int(t_max/dt)-1
umin = np.array([-T_hover])
umax = np.array([30.-T_hover])
xmin=np.array([ground_altitude, -5.])
xmax=np.array([10., 5.])
set_pt = ground_altitude+0.1
ref = np.array([[set_pt for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])
ctrl_pert_var = 0.5

# Filter Parameters:
eta = 0.6**2 # measurement covariance
Nb = 3 # number of ensemble
nk = 5 # number of steps for multi-step prediction
E = np.array([0,-gravity*mass])
B_ensemble = np.stack([B_mean-np.array([[0.],[0.6]]), B_mean, B_mean+np.array([[0.],[0.6]])],axis=2)
#B_ensemble_list = [B_mean-np.array([[0.],[0.5]]), B_mean, B_mean+np.array([[0.],[0.5]])]
true_sys = LinearSystemDynamics(A, B_mean)

# EDMD Parameters:
tune_edmd = False
l1_edmd = 0.06755207638157183                    # l1 regularization strength for position states
l1_ratio_edmd = 1.0                               # l1-l2 ratio for position states
n_monominals = 10
n_lift = n_monominals+1


E_edmd = np.array([0,-gravity*mass])
E_edmd = np.concatenate((E_edmd, np.zeros(n_lift)))
B_mean_edmd = np.concatenate((B_mean, np.zeros((n_lift, Nu))))
B_ensemble_edmd = np.stack([B_mean-np.array([[0.],[0.6]]), B_mean, B_mean+np.array([[0.],[0.6]])],axis=2)
B_ensemble_edmd = np.stack([np.concatenate((B_ensemble_edmd[:,:,0], -0.1*np.ones((n_lift,Nu)))),
                                B_mean_edmd,
                                np.concatenate((B_ensemble_edmd[:,:,2], 0.1*np.ones((n_lift,Nu))))], axis=2)
A_edmd = np.zeros((n_lift+Ns, n_lift+Ns))
A_edmd[:Ns,:Ns] = A
C_edmd = np.zeros((Ns, n_lift+Ns))
C_edmd[:,:Ns] = np.eye(Ns)
K_p, K_d = [[25.125]], [[10.6331]]

#%%
#! ===============================================   COLLECT DATA    ================================================
x_ep, xd_ep, u_ep, u_nom_ep, traj_ep, B_ep, mpc_cost_ep, t_ep = [], [], [], [], [], [], [], []
B_ep_edmd = []
#x_ep_edmd, xd_ep_keedmd, u_ep_keedmd, u_nom_ep_keedmd, traj_ep_keedmd, B_ep_keedmd, mpc_cost_ep_keedmd, t_ep_keedmd = [], [], [], [], [], [], [], []
#x_th, u_th  = [], []
B_ep.append(B_ensemble) # B_ep[N_ep] of numpy array [Ns,Nu,Ne]
B_ep_edmd.append(B_ensemble_edmd) # B_ep[N_ep] of numpy array [Ns+Neig,Nu,Ne]

# Define controller for data collection:
lin_dyn = LinearSystemDynamics(A, B_ep[-1][:, :, 1])
controller = RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ep[-1], D=Dmatrix,
                            noise_var=ctrl_pert_var)
controller_nom = RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ep[-1],
                                D=Dmatrix, noise_var=0.)
for ep in range(N_ep):

    print(f"Episode {ep}")
    # Design robust MPC with current ensemble of Bs and execute experiment state space model:
    # TODO: Sample B (m) and modify true system to add variation in the data
    x_tmp, u_tmp = system.simulate(z_0, controller, t_eval)
    u_nom_tmp = np.array([controller_nom.eval(x_tmp[ii,:], t_eval[ii]) for ii in range(x_tmp.shape[0]-1)])
    #x_th_tmp, u_th_tmp = controller.get_thoughts_traj()
    #x_th.append(x_th_tmp) # x_th[Nep][Nt][Ne] [Ns,Np]_NumpyArray
    #u_th.append(u_th_tmp)  # u_th [Nep][Nt] [Nu,Np]_NumpyArray
    x_ep.append(x_tmp.T) # x_ep [Nep][Nt+1] [Ns,]_NumpyArray
    xd_ep.append(ref.tolist())
    u_ep.append(u_tmp.T) # u_ep [Nep][Nt] [Nu,]_NumpyArray
    u_nom_ep.append(u_nom_tmp)  # u_ep [Nep][Nt] [Nu,]_NumpyArray
    t_ep.append(t_eval.tolist()) # t_ep [Nep][Nt+1,]_NumpyArray
    mpc_cost_ep.append(np.sum(np.diag((x_tmp[:-1,:].T-ref[:,:-1]).T@Q@(x_tmp[:-1,:].T-ref[:,:-1]) + u_tmp@R@u_tmp.T)))

#%%
#! ===============================================   LEARN MODELS    ================================================

# Update the ensemble of Bs with inverse Kalman filter of state space model:
print('Learning ensemble of B, state space model')
inverse_kalman_filter = InverseKalmanFilter(A, B_mean, E, eta, B_ensemble, dt, nk)
inverse_kalman_filter.fit(x_ep, u_ep)
B_ep.append(inverse_kalman_filter.B_ensemble)

# Construct Koopman eigenfunctions and learn KEEDMD model:
print('Learning EDMD model')
x_arr, xd_arr, u_arr, u_nom_arr, t_arr = np.array(x_ep), np.array(xd_ep), np.array(u_ep), np.array(u_nom_ep), np.array(t_ep)
x_arr, xd_arr, u_arr, u_nom_arr = np.swapaxes(x_arr,1,2), np.swapaxes(xd_arr,1,2), np.swapaxes(u_arr,1,2), np.swapaxes(u_nom_arr,1,2)

monomial_basis = Monomials(2,10)
monomial_basis.construct_basis()
edmd_model = Edmd(basis=monomial_basis, system_dim=2, control_dim=1, l1=l1_edmd, l1_ratio=l1_ratio_edmd, add_ones=True, add_state=True)
X, X_d, Z, Z_dot, U, U_nom, t = edmd_model.process(x_arr, xd_arr, u_arr, u_nom_arr, t_arr)

if tune_edmd:
    print('Tuning EDMD regularization and fitting the EDMD model')
    edmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom)
else:
    print('Fitting the EDMD model')
    edmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)

# Update the ensemble of Bs with inverse Kalman filter of lifted model:
print('Learning ensemble of Bs for lifted state space model')
A_edmd = edmd_model.A
inverse_kalman_filter_edmd = InverseKalmanFilter(A_edmd, B_mean_edmd, E_edmd, eta, B_ensemble_edmd, dt, nk)
z_ep = [edmd_model.lift(np.array(x), np.array(xd)).T for x, xd in zip(x_ep, xd_ep)]
inverse_kalman_filter_edmd.fit(z_ep, u_ep)
B_ep_edmd.append(inverse_kalman_filter_edmd.B_ensemble)

#%%
#! =====================================   EVALUATE CLOSED LOOP PERFORMANCE    ========================================
# Evaluate closed loop for state space model:
print('Mean B: ', B_mean)
print('Ensemble used in control design: ', B_ep[-1])
dynamics_ss = lin_dyn = LinearSystemDynamics(A, B_ep[-1][:, :, 1])
controller_ss = RobustMpcDense(dynamics_ss, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ep[-1], D=Dmatrix, gather_thoughts=False)
x_ss_val, u_ss_val = system.simulate(z_0, controller_ss, t_eval)

# Evaluate closed loop for KEEDMD model:
print('Mean B: ', edmd_model.B)
print('Ensemble used in control design: ', B_ep_edmd[-1])
dynamics_keedmd = LinearSystemDynamics(A_edmd, B_ep_edmd[-1][:, :, 1])
controller_keedmd = RobustMpcDense(dynamics_keedmd, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ep_edmd[-1], D=Dmatrix, edmd_object=edmd_model, gather_thoughts=False)
x_keedmd_val, u_keedmd_val = system.simulate(z_0, controller_keedmd, t_eval)

#%%
#! ===============================================   PLOT RESULTS    =================================================
def plot_ss_keedmd_comparison(t_eval, x_ss_val, u_ss_val, x_keedmd_val, u_keedmd_val, ref, ground_altitude, T_hover):
    plt.figure(figsize=(6,5))

    plt.subplot(2,2,1)
    plt.plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2, label='Ground')
    plt.plot(t_eval, ref[0,:], '--g', lw=2, label='Set point')
    plt.plot(t_eval, x_ss_val[:, 0], label='z')
    plt.fill_between(t_eval, ref[0,:], x_ss_val[:,0], alpha=0.2)

    err_norm = (t_eval[-1]-t_eval[0])*np.sum(np.square(x_ss_val[:, 0].T - ref[0,:]))/x_ss_val[:, 0].shape[0]
    plt.text(1.2, 0.5, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))

    plt.title('State space model')
    #plt.xlabel('Time (sec)')
    plt.ylabel('z (m)')
    plt.grid()

    plt.subplot(2,2,3)
    plt.plot(t_eval[:-1], u_ss_val[:, 0], label='Thrust')
    plt.plot([t_eval[0], t_eval[-2]], [umax + T_hover, umax + T_hover], '--k', lw=2, label='Max thrust')
    plt.fill_between(t_eval[:-1], np.zeros_like(u_ss_val[:,0]), u_ss_val[:,0], alpha=0.2)
    ctrl_norm = (t_eval[-2] - t_eval[0]) * np.sum(
        (np.square(u_ss_val[:, 0])) / u_ss_val[:, 0].shape[0])
    plt.text(1.2, 11, "$\int u_n^2=${0:.2f}".format(ctrl_norm))
    plt.xlabel('Time (sec)')
    plt.ylabel('Thrust (N)')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2, label='Ground')
    plt.plot(t_eval, ref[0, :], '--g', lw=2, label='Set point')
    plt.plot(t_eval, x_keedmd_val[:, 0], label='z')
    plt.fill_between(t_eval, ref[0, :], x_keedmd_val[:, 0], alpha=0.2)

    err_norm = (t_eval[-1] - t_eval[0]) * np.sum(np.square(x_keedmd_val[:, 0].T - ref[0, :])) / x_keedmd_val[:, 0].shape[0]
    plt.text(1.2, 0.5, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))

    plt.title('Lifted space model')
    #plt.xlabel('Time (sec)')
    #plt.ylabel('z (m)')
    plt.grid()
    plt.legend(loc='upper right')

    plt.subplot(2, 2, 4)
    plt.plot(t_eval[:-1], u_keedmd_val[:, 0], label='T')
    plt.plot([t_eval[0], t_eval[-2]], [umax + T_hover, umax + T_hover], '--k', lw=2, label='Max thrust')
    plt.fill_between(t_eval[:-1], np.zeros_like(u_keedmd_val[:, 0]), u_keedmd_val[:, 0], alpha=0.2)
    ctrl_norm = (t_eval[-2] - t_eval[0]) * np.sum(
        (np.square(u_keedmd_val[:, 0])) / u_keedmd_val[:, 0].shape[0])
    plt.text(1.2, 11, "$\int u_n^2=${0:.2f}".format(ctrl_norm))
    plt.xlabel('Time (sec)')
    #plt.ylabel('Thrust (N)')
    plt.grid()
    plt.legend(loc='upper right')


    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.tight_layout()
    plt.savefig('core/examples/results/edmd_comparison.pdf', format='pdf', dpi=2400)
    plt.show()

sp.io.savemat('./core/examples/1d_drone_edmd.mat', {'t_eval':t_eval,
                                                    'x_ss_val':x_ss_val,
                                                    'u_ss_val':u_ss_val,
                                                    'x_keedmd_val':x_keedmd_val,
                                                    'u_keedmd_val':u_keedmd_val,
                                                    'ref':ref,
                                                    'ground_altitude':ground_altitude,
                                                    'T_hover':T_hover})

plot_ss_keedmd_comparison(t_eval, x_ss_val, u_ss_val, x_keedmd_val, u_keedmd_val, ref, ground_altitude, T_hover)