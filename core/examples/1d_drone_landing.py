from ..systems import OneDimDrone, LinearOneDimDrone
from ..controllers import RobustMpcDense, MPCController, OpenLoopController
from ..dynamics import SystemDynamics, LinearSystemDynamics
from ..learning import InverseKalmanFilter

from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy as sp

# %%
print("Starting 1D Drone Landing Simulation..")
# ! ===============================================   SET PARAMETERS    ================================================
# Define system parameters of the drone:
mass = 1  # Drone mass (kg)
rotor_rad = 0.08  # Rotor radius (m)
drag_coeff = 0.5  # Drag coefficient
air_dens = 1.25  # Air density (kg/m^3)
area = 0.04  # Drone surface area in xy-plane (m^2)
gravity = 9.81  # Gravity (m/s^2)
T_hover = mass * gravity  # Hover thrust (N)
ground_altitude = 0.2  # Altitude corresponding to drone landed (m)
# system = OneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)
system = LinearOneDimDrone(mass, rotor_rad, drag_coeff, air_dens, area, gravity, ground_altitude, T_hover)

# Define initial linearized model and ensemble of Bs (linearized around hover):
A = np.array([[0., 1.], [0., 0.]])
B_mean = np.array([[0.], [1 / mass]])
Ns = B_mean.shape[0]
Nu = B_mean.shape[1]

# Define simulation parameters:
z_0 = np.array([4., 0.])  # Initial position
dt = 1e-2  # Time step length
t_max = 1.5  # End time (sec)
t_eval = np.linspace(0, t_max, int(t_max / dt))  # Simulation time points
N_ep = 6  # Number of episodes

# Model predictive controller parameters:
Q = np.array([[2e4, 0.], [0., 1.]])
QN = Q
R = np.array([[1]])
Dmatrix = sp.sparse.diags([1e6,1000])
N_steps = int(t_max/dt)-1
umin = np.array([-T_hover])
umax = np.array([30.-T_hover])
xmin=np.array([ground_altitude, -5.])
xmax=np.array([10., 5.])
ref = np.array([[ground_altitude+0.05 for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])

#! Filter Parameters:
eta = 0.6**2 # measurement covariance
Nb = 3 # number of ensemble
nk = 5 # number of steps for multi-step prediction
# B_ensemble = np.zeros((Ns,Nu,Nb))
# for i in range(Nb):
#     B_ensemble[:,:,i] = B_mean+np.array([[0.],[np.random.uniform(-0.5,0.5)]])

E = np.array([0, -gravity * mass])
B_ensemble = np.stack([B_mean - np.array([[0.], [0.6]]), B_mean, B_mean + np.array([[0.], [0.6]])], axis=2)

# B_ensemble_list = [B_mean-np.array([[0.],[0.5]]), B_mean, B_mean+np.array([[0.],[0.5]])]
true_sys = LinearSystemDynamics(A, B_mean)
print(f"Main parameters: Nb:{Nb}, N_ep:{N_ep}, N_t:{N_steps}")

# %%
# ! ===============================================   RUN LEARNING B EXPERIMENT    ================================================

print(f"Run RUN LEARNING B EXPERIMENT")
inverse_kalman_filter = InverseKalmanFilter(A, B_mean, E, eta, B_ensemble, dt, nk)

x_ep, xd_ep, u_ep, traj_ep, B_ep, mpc_cost_ep, t_ep = [], [], [], [], [], [], []
x_th, u_th = [], []
# B_ensemble [Ns,Nu,Ne] numpy array
B_ep.append(B_ensemble)  # B_ep[N_ep] of numpy array [Ns,Nu,Ne]

for ep in range(N_ep):
    print(f"Episode {ep}")

    # Design robust MPC with current ensemble of Bs and execute experiment:
    lin_dyn = LinearSystemDynamics(A, B_ep[-1][:, :, 1])
    controller = RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ep[-1],
                                D=Dmatrix, gather_thoughts=True)
    x_tmp, u_tmp = system.simulate(z_0, controller, t_eval)
    x_th_tmp, u_th_tmp = controller.get_thoughts_traj()
    x_th.append(x_th_tmp)  # x_th[Nep][Nt][Ne] [Ns,Np]_NumpyArray
    u_th.append(u_th_tmp)  # u_th [Nep][Nt] [Nu,Np]_NumpyArray
    x_ep.append(x_tmp.T)  # x_ep [Nep][Ns, Nt+1]_NumpyArray
    xd_ep.append(np.transpose(ref).tolist())
    u_ep.append(u_tmp.T)  # u_ep [Nep][Nu,Nt]_NumpyArray
    t_ep.append(t_eval.tolist())  # t_ep [Nep][Nt+1,]_NumpyArray
    mpc_cost_ep.append(
        np.sum(np.diag((x_tmp[:-1, :].T - ref[:, :-1]).T @ Q @ (x_tmp[:-1, :].T - ref[:, :-1]) + u_tmp @ R @ u_tmp.T)))
    if ep == N_ep - 1:
        break

    print('MPC Cost: ', mpc_cost_ep[-1])

    # Update the ensemble of Bs with inverse Kalman filter:
    # x_flat, xd_flat, xdot_flat, u_flat, t_flat = inverse_kalman_filter.process(np.array(x_ep), np.array(xd_ep),
    #                                                                           np.array(u_ep), np.array(t_ep))
    #if (ep > 2):
    inverse_kalman_filter.fit(x_ep, u_ep) 
    B_ep.append(inverse_kalman_filter.B_ensemble)

x_ep, xd_ep, u_ep, traj_ep, B_ep, t_ep = np.array(x_ep), np.array(xd_ep), np.array(u_ep), np.array(traj_ep), \
                                         np.array(B_ep), np.array(t_ep)


# %%
# ! ===============================================   PLOT RESULTS    =================================================

# Plot evolution of ensembles of B and predicted trajectories for each episode:

def plot_summary_EnMPC(B_ep, N_ep, mpc_cost_ep, t_eval, x_ep, u_ep, x_th, u_th, ground_altitude,T_hover):
    f2 = plt.figure(figsize=(15,12))
    gs2 = gridspec.GridSpec(5,3, figure=f2)
    
    # - Plot evolution of B ensemble:
    n_B = B_ep[0].shape[2]
    x_ensemble, y_ensemble = [], []
    x_ep_plt, y_min, y_max = [], [], []
    for ep in range(N_ep):
        x_ep_plt.append(ep)
        y_min.append(B_ep[ep][1, 0, 0])
        # print(f"min {B_ep[ep][1,0,0]}, max {B_ep[ep][1,0,n_B-1]}")
        y_max.append(B_ep[ep][1, 0, n_B - 1])  # B_ep[N_ep] of numpy array [Ns,Nu,Ne]
        for ii in range(n_B):
            x_ensemble.append(ep)
            y_ensemble.append(B_ep[ep][1, 0, ii])

    a0 = f2.add_subplot(gs2[0, :])
    a0.scatter(x_ensemble, y_ensemble)
    a0.fill_between(x_ep_plt, y_min, y_max, color='b', alpha=0.1)
    a0.set_title('Values of Bs in the Ensemble at Each Episode')
    a0.set_xlabel('Episode')
    a0.set_ylabel('B value')
    a0.xaxis.set_major_locator(MaxNLocator(integer=True))
    a0.grid()

    # - Plot predicted trajectories for 3 selected episodes:
    # Plot MPC cost and executed trajectory every episode:
    plot_ep = [0, int((N_ep - 1) / 2), N_ep - 1]

    # - Plot evolution of MPC cost:
    y_mpc = [c / mpc_cost_ep[0] for c in mpc_cost_ep]
    b0 = f2.add_subplot(gs2[1, :])
    b0.plot(x_ep_plt, y_mpc)
    b0.set_title('MPC Cost Evolution')
    b0.set_xlabel('Episode')
    b0.set_ylabel('Normalized MPC Cost')
    b0.xaxis.set_major_locator(MaxNLocator(integer=True))
    b0.grid()

    # - Plot executed trajectories and control effort for each episode:
    pos_plot, u_plot, vel_plot = [], [], []
    N_e_summary = 3
    for ii in range(N_e_summary):
        pos_plot.append(f2.add_subplot(gs2[2, ii]))
        pos_plot[ii].plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2,
                          label='Ground constraint')
        pos_plot[ii].plot(t_eval, x_ep[plot_ep[ii], 0, :], label='z')

        Nt = len(x_th[0])
        ensemble_time_indices = [0, int(Nt / 4), int(Nt / 3)]
        for index_time_ensemble in ensemble_time_indices:
            for x_th_en in x_th[plot_ep[ii]][index_time_ensemble]:  # for every ensemble at time zero
                pos_plot[ii].plot(
                    t_eval + t_eval[index_time_ensemble],
                    np.hstack([x_ep[plot_ep[ii]][0, index_time_ensemble], x_th_en[0, :]]), lw=0.5, c='k', alpha=0.5)
                pos_plot[ii].scatter(t_eval[index_time_ensemble], x_ep[plot_ep[ii]][0, index_time_ensemble], alpha=0.5,
                                     color='r', s=20)

        # b1_lst[ii].fill_between(t_eval, ref[0,:], x_ep[plot_ep[ii], :, 0], alpha=0.2)
        err_norm = (t_eval[-1] - t_eval[0]) * np.sum(np.square(x_ep[plot_ep[ii], 0, :].T - ref[0, :])) / \
                   x_ep[plot_ep[ii], 0, :].shape[0]
        # b1_lst[ii].text(1.2, 0.5, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))

        pos_plot[ii].set_title('Executed trajectory, ep ' + str(plot_ep[ii]))
        # pos_plot[ii].set_xlabel('Time (sec)')
        pos_plot[ii].set_ylabel('z (m)')
        pos_plot[ii].grid()

        vel_plot.append(f2.add_subplot(gs2[3, ii]))
        vel_plot[ii].plot(t_eval, x_ep[plot_ep[ii], 1, :], label='$\dot{z}$')

        for index_time_ensemble in ensemble_time_indices:
            for x_th_en in x_th[plot_ep[ii]][index_time_ensemble]:  # for every ensemble at time zero
                vel_plot[ii].plot(
                    t_eval + t_eval[index_time_ensemble],
                    np.hstack([x_ep[plot_ep[ii]][1, index_time_ensemble], x_th_en[1, :]]), lw=0.5, c='k', alpha=0.5)
                vel_plot[ii].scatter(t_eval[index_time_ensemble], x_ep[plot_ep[ii]][1, index_time_ensemble], alpha=0.5,
                                     color='r', s=20)
        vel_plot[ii].set_ylabel('$\dot{z}$ (m/s)')
        vel_plot[ii].grid()

        print(f"heyd")

        u_plot.append(f2.add_subplot(gs2[4, ii]))
        u_plot[ii].plot(t_eval[:-1], u_ep[plot_ep[ii], 0, :], label='T')
        u_plot[ii].plot([t_eval[0], t_eval[-2]], [umax + T_hover, umax + T_hover], '--r', lw=2, label='Max thrust')

        for index_time_ensemble in ensemble_time_indices:
            u_plot[ii].plot(
                t_eval[:-1] + t_eval[index_time_ensemble], T_hover + u_th[plot_ep[ii]][index_time_ensemble][0, :],
                lw=0.5, c='k', alpha=0.5)
            u_plot[ii].scatter(t_eval[index_time_ensemble], u_ep[plot_ep[ii]][0, index_time_ensemble], alpha=0.5,
                               color='r', s=20)

        u_plot[ii].fill_between(t_eval[:-1], np.zeros_like(u_ep[plot_ep[ii], 0, :]), u_ep[plot_ep[ii], 0, :], alpha=0.2)
        ctrl_norm = (t_eval[-2] - t_eval[0]) * np.sum(
            (np.square(u_ep[plot_ep[ii], 0, :])) / u_ep[plot_ep[ii], 0, :].shape[0])
        u_plot[ii].text(1.2, 11, "$\int u_n^2=${0:.2f}".format(ctrl_norm))
        u_plot[ii].set_title('Executed control effort, ep ' + str(plot_ep[ii]))
        u_plot[ii].set_xlabel('Time (sec)')
        u_plot[ii].set_ylabel('Thrust (N)')
        u_plot[ii].grid()
    pos_plot[-1].legend(loc='lower right')
    u_plot[-1].legend(loc='upper right')
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    gs2.tight_layout(f2)
    f2.savefig('core/examples/results/executed_traj.pdf', format='pdf', dpi=2400)
    plt.show()

    sp.io.savemat('./core/examples/1d_drone.mat', {'B_ep': B_ep,
                                                   'N_ep': N_ep,
                                                   'mpc_cost_ep': mpc_cost_ep,
                                                   't_eval': t_eval,
                                                   'x_ep': x_ep,
                                                   'u_ep': u_ep,
                                                   'x_th': x_th,
                                                   'u_th': u_th,
                                                   'ground_altitude': ground_altitude,
                                                   'T_hover': T_hover})


plot_summary_EnMPC(B_ep, N_ep, mpc_cost_ep, t_eval, x_ep, u_ep, x_th, u_th, ground_altitude, T_hover)
