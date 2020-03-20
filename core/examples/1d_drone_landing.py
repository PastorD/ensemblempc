from ..systems import OneDimDrone, LinearOneDimDrone
from ..controllers import RobustMpcDense, MPCController, OpenLoopController
from ..dynamics import SystemDynamics, LinearSystemDynamics
from ..learning import InverseKalmanFilter, KoopmanEigenfunctions, Keedmd

from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
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
N_ep = 5                                                   # Number of episodes

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
set_pt = ground_altitude+0.1
ref = np.array([[set_pt for _ in range(N_steps+1)],
                [0. for _ in range(N_steps+1)]])

# Filter Parameters:
eta = 0.6**2 # measurement covariance
Nb = 3 # number of ensemble
nk = 5 # number of steps for multi-step prediction
E = np.array([0,-gravity*mass])
B_ensemble = np.stack([B_mean-np.array([[0.],[0.6]]), B_mean, B_mean+np.array([[0.],[0.6]])],axis=2)
#B_ensemble_list = [B_mean-np.array([[0.],[0.5]]), B_mean, B_mean+np.array([[0.],[0.5]])]
true_sys = LinearSystemDynamics(A, B_mean)

# KEEDMD Parameters:
# - Koopman eigenfunction parameters
eigenfunction_max_power = 3                             # Max power of variables in eigenfunction products
l2_diffeomorphism = 0.0                                 # l2 regularization strength
jacobian_penalty_diffeomorphism = 1e1                   # Estimator jacobian regularization strength
diff_n_epochs = 100                                     # Number of epochs
diff_train_frac = 0.9                                   # Fraction of data to be used for training
diff_n_hidden_layers = 2                                # Number of hidden layers
diff_layer_width = 25                                   # Number of units in each layer
diff_batch_size = 8                                     # Batch size
diff_learn_rate = 0.06842                               # Leaning rate
diff_learn_rate_decay = 0.95                            # Learning rate decay
diff_dropout_prob = 0.25                                # Dropout rate

# - EDMD Regularization Parameters:
tune_keedmd = False
l1_pos_keedmd = 6.90225469066697e-07                    # l1 regularization strength for position states
l1_pos_ratio_keedmd = 1.0                               # l1-l2 ratio for position states
l1_vel_keedmd = 0.03392536629651686                     # l1 regularization strength for velocity states
l1_vel_ratio_keedmd = 1.0                               # l1-l2 ratio for velocity states
l1_eig_keedmd = 1e-15                                   # l1 regularization strength for eigenfunction states
l1_eig_ratio_keedmd = 1                                 # l1-l2 ratio for eigenfunction states

Neig = (eigenfunction_max_power+1)**Ns
E_keedmd = np.array([0,-gravity*mass])
E_keedmd = np.concatenate((E_keedmd, np.zeros(Neig)))
B_mean_keedmd = np.concatenate((B_mean, np.zeros((Neig, Nu))))
B_ensemble_keedmd = np.stack([B_mean-np.array([[0.],[0.6]]), B_mean, B_mean+np.array([[0.],[0.6]])],axis=2)
B_ensemble_keedmd = np.stack([np.concatenate((B_ensemble_keedmd[:,:,0], np.zeros((Neig,Nu)))),
                                B_mean_keedmd,
                                np.concatenate((B_ensemble_keedmd[:,:,2], np.zeros((Neig,Nu))))], axis=2)
A_keedmd = np.zeros((Neig+Ns, Neig+Ns))
A_keedmd[:Ns,:Ns] = A
C_keedmd = np.zeros((Ns, Neig+Ns))
C_keedmd[:,:Ns] = np.eye(Ns)
K_p, K_d = [[25.125]], [[10.6331]]

#! ================================================ Run limited MPC Controller ===========================================

lin_dyn_mean = LinearSystemDynamics(A, B_mean)
ctrl_tmp_mean = RobustMpcDense(lin_dyn_mean, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ensemble, D=Dmatrix)
#lin_dyn_b = [ LinearSystemDynamics(A, B_ensemble[:,:,i]) for i in range(Nb)]
#ctrl_tmp_b = [ RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref) for lin_dyn in lin_dyn_b]

ctrl_tmp_mean.eval(z_0, 0)
u_mean = ctrl_tmp_mean.get_control_prediction()
z_mean = ctrl_tmp_mean.get_state_prediction()
z_b = ctrl_tmp_mean.get_ensemble_state_prediction()

t_z = np.linspace(0,ctrl_tmp_mean.N*dt,ctrl_tmp_mean.N)
f,axarr=plt.subplots(3, sharex=True)
f.subplots_adjust(hspace=.1)
#plt.figure(figsize=(12,6))
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
#plt.show()


#%%
#! ===============================================   RUN EXPERIMENT    ================================================
inverse_kalman_filter = InverseKalmanFilter(A, B_mean, E, eta, B_ensemble, dt, nk)
inverse_kalman_filter_keedmd = InverseKalmanFilter(A_keedmd, B_mean_keedmd, E_keedmd, eta, B_ensemble_keedmd, dt, nk)

A_cl = A - np.dot(B_mean, np.concatenate((K_p, K_d),axis=1))
BK = np.dot(B_mean, np.concatenate((K_p, K_d),axis=1))
eigenfunction_basis = KoopmanEigenfunctions(n=Ns, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
eigenfunction_basis.build_diffeomorphism_model(jacobian_penalty=jacobian_penalty_diffeomorphism, n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width, batch_size= diff_batch_size, dropout_prob=diff_dropout_prob)

x_ep, xd_ep, u_ep, traj_ep, B_ep, mpc_cost_ep, t_ep = [], [], [], [], [], [], []
x_ep_keedmd, xd_ep_keedmd, u_ep_keedmd, traj_ep_keedmd, B_ep_keedmd, mpc_cost_ep_keedmd, t_ep_keedmd = [], [], [], [], [], [], []
x_th, u_th  = [], []
# B_ensemble [Ns,Nu,Ne] numpy array
B_ep.append(B_ensemble) # B_ep[N_ep] of numpy array [Ns,Nu,Ne]
B_ep_keedmd.append(B_ensemble_keedmd) # B_ep[N_ep] of numpy array [Ns+Neig,Nu,Ne]

for ep in range(N_ep):
    print(f"Episode {ep}")
    # Calculate predicted trajectories for each B in the ensemble:
    # traj_ep_tmp = []
    # for i in range(Nb):
    #     lin_dyn = LinearSystemDynamics(A, B_ensemble[:,:,i])
    #     ctrl_tmp = RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref)
    #     ctrl_tmp.eval(z_0, 0)
    #     traj_ep_tmp.append(ctrl_tmp.get_state_prediction())
    # traj_ep.append(traj_ep_tmp)

    # Design robust MPC with current ensemble of Bs and execute experiment state space model:
    lin_dyn = LinearSystemDynamics(A, B_ep[-1][:,:,1])
    controller = RobustMpcDense(lin_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ensemble, D=Dmatrix)
    x_tmp, u_tmp = system.simulate(z_0, controller, t_eval) 
    x_th_tmp, u_th_tmp = controller.get_thoughts_traj()
    x_th.append(x_th_tmp) # x_th[Nep][Nt][Ne] [Ns,Np]_NumpyArray
    u_th.append(u_th_tmp)  # u_th [Nep][Nt] [Nu,Np]_NumpyArray
    x_ep.append(x_tmp) # x_ep [Nep][Nt+1] [Ns,]_NumpyArray
    xd_ep.append(np.transpose(ref).tolist())
    u_ep.append(u_tmp) # u_ep [Nep][Nt] [Nu,]_NumpyArray
    t_ep.append(t_eval.tolist()) # t_ep [Nep][Nt+1,]_NumpyArray
    mpc_cost_ep.append(np.sum(np.diag((x_tmp[:-1,:].T-ref[:,:-1]).T@Q@(x_tmp[:-1,:].T-ref[:,:-1]) + u_tmp@R@u_tmp.T)))

    # Design robust MPC with current ensemble of Bs and execute experiment with KEEDMD model:
    if ep > 0:
        lifted_dyn = LinearSystemDynamics(A_keedmd, B_ep_keedmd[-1][:, :, 1])
        controller_keedmd = RobustMpcDense(lifted_dyn, N_steps, dt, umin, umax, xmin, xmax, Q, R, QN, ref, ensemble=B_ep_keedmd[-1], D=Dmatrix, edmd_object=keedmd_model, gather_thoughts=False)
        #z_0_keedmd = keedmd_model.lift(z_0.reshape(z_0.shape[0],1), np.array([[set_pt],[0.]])).squeeze()
        x_keedmd_tmp, u_keedmd_tmp = system.simulate(z_0, controller_keedmd, t_eval)
        #x_keedmd_tmp = np.dot(keedmd_model.C, z_keedmd_tmp)
        x_ep_keedmd.append(x_keedmd_tmp)  # x_ep [Nep][Nt+1] [Ns,]_NumpyArray
        xd_ep_keedmd.append(np.transpose(ref).tolist())
        u_ep_keedmd.append(u_keedmd_tmp)  # u_ep [Nep][Nt] [Nu,]_NumpyArray
        t_ep_keedmd.append(t_eval.tolist())  # t_ep [Nep][Nt+1,]_NumpyArray
        mpc_cost_ep_keedmd.append(
            np.sum(np.diag((x_keedmd_tmp[:-1, :].T - ref[:, :-1]).T @ Q @ (x_keedmd_tmp[:-1, :].T - ref[:, :-1]) + u_keedmd_tmp @ R @ u_keedmd_tmp.T)))
    else:
        x_keedmd_tmp, u_keedmd_tmp = x_tmp, u_tmp
        x_ep_keedmd.append(x_keedmd_tmp)  # x_ep [Nep][Nt+1] [Ns,]_NumpyArray
        xd_ep_keedmd.append(np.transpose(ref).tolist())
        u_ep_keedmd.append(u_keedmd_tmp)  # u_ep [Nep][Nt] [Nu,]_NumpyArray
        t_ep_keedmd.append(t_eval.tolist())  # t_ep [Nep][Nt+1,]_NumpyArray
        mpc_cost_ep_keedmd.append(
            np.sum(np.diag((x_keedmd_tmp[:-1, :].T - ref[:, :-1]).T @ Q @ (
                        x_keedmd_tmp[:-1, :].T - ref[:, :-1]) + u_keedmd_tmp @ R @ u_keedmd_tmp.T)))

    if ep == N_ep-1:
        if tune_keedmd:
            print('Tuning KEEDMD regularization...')
            x_arr_keedmd, xd_arr_keedmd, u_arr_keedmd, t_arr_keedmd = np.array(x_ep_keedmd), np.array(
                xd_ep_keedmd), np.array(u_ep_keedmd), np.array(t_ep_keedmd)
            eigenfunction_basis.fit_diffeomorphism_model(X=x_arr_keedmd, t=t_arr_keedmd, X_d=xd_arr_keedmd,
                                                         l2=l2_diffeomorphism,
                                                         learning_rate=diff_learn_rate,
                                                         learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs,
                                                         train_frac=diff_train_frac, batch_size=diff_batch_size,
                                                         verbose=False)
            eigenfunction_basis.construct_basis(ub=xmax, lb=xmin)
            keedmd_model = Keedmd(eigenfunction_basis, Ns, Nu, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd,
                                  l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd,
                                  l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d, add_state=True)
            X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(x_arr_keedmd, xd_arr_keedmd, u_arr_keedmd, u_arr_keedmd,
                                                                 t_arr_keedmd)
            keedmd_model.tune_fit(X, X_d, Z, Z_dot, U, U_nom)
        break

    # Update the ensemble of Bs with inverse Kalman filter of state space model:
    print('- Updating ensemble, state space model')
    x_flat, xd_flat, xdot_flat, u_flat, t_flat = inverse_kalman_filter.process(np.array(x_ep), np.array(xd_ep),
                                                                               np.array(u_ep), np.array(t_ep))
    inverse_kalman_filter.fit(x_flat, xdot_flat, u_flat) 
    B_ep.append(inverse_kalman_filter.B_ensemble)

    # Update the ensemble of Bs with inverse Kalman filter of lifted model:
    print('- Updating A-matrix, lifted model')
    x_arr_keedmd, xd_arr_keedmd, u_arr_keedmd, t_arr_keedmd = np.array(x_ep_keedmd), np.array(xd_ep_keedmd), np.array(u_ep_keedmd), np.array(t_ep_keedmd)
    eigenfunction_basis.fit_diffeomorphism_model(X=x_arr_keedmd, t=t_arr_keedmd, X_d=xd_arr_keedmd, l2=l2_diffeomorphism,
                                                 learning_rate=diff_learn_rate,
                                                 learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs,
                                                 train_frac=diff_train_frac, batch_size=diff_batch_size, verbose=False)
    eigenfunction_basis.construct_basis(ub=xmax, lb=xmin)
    keedmd_model = Keedmd(eigenfunction_basis, Ns, Nu, l1_pos=l1_pos_keedmd, l1_ratio_pos=l1_pos_ratio_keedmd,
                          l1_vel=l1_vel_keedmd, l1_ratio_vel=l1_vel_ratio_keedmd, l1_eig=l1_eig_keedmd,
                          l1_ratio_eig=l1_eig_ratio_keedmd, K_p=K_p, K_d=K_d, add_state=True)
    X, X_d, Z, Z_dot, U, U_nom, t = keedmd_model.process(x_arr_keedmd, xd_arr_keedmd, u_arr_keedmd, u_arr_keedmd, t_arr_keedmd)
    keedmd_model.fit(X, X_d, Z, Z_dot, U, U_nom)
    # TODO: Add exploration noise and train KEEDMD based on that noise (now the control effect on the eigenfuncs are trained based on zero control input)

    print('- Updating ensemble, lifted model')
    A_keedmd = keedmd_model.A
    inverse_kalman_filter_keedmd.A = keedmd_model.A
    inverse_kalman_filter_keedmd.fit(Z, Z_dot, U)
    B_ep_keedmd.append(inverse_kalman_filter_keedmd.B_ensemble)

x_ep, xd_ep, u_ep, traj_ep, B_ep, t_ep = np.array(x_ep), np.array(xd_ep), np.array(u_ep), np.array(traj_ep), \
                                         np.array(B_ep), np.array(t_ep)
x_ep_keedmd, xd_ep_keedmd, u_ep_keedmd, traj_ep_keedmd, B_ep_keedmd, t_ep_keedmd = np.array(x_ep_keedmd), np.array(xd_ep_keedmd), np.array(u_ep_keedmd), np.array(traj_ep_keedmd), \
                                         np.array(B_ep_keedmd), np.array(t_ep_keedmd)

# TODO: Plot summary_EnMPC (For now, discuss what's the best way to show data later)

#%%
#! ===============================================   PLOT RESULTS    =================================================

# Plot evolution of ensembles of B and predicted trajectories for each episode:

def plot_summary_EnMPC(B_ep, N_ep, mpc_cost_ep, t_eval, x_ep, u_ep, x_th, u_th, ground_altitude,T_hover):
    f2 = plt.figure(figsize=(18,9))
    gs2 = gridspec.GridSpec(5,3, figure=f2)
    
    # - Plot evolution of B ensemble:
    n_B = B_ep[0].shape[2]
    x_ensemble, y_ensemble = [], []
    x_ep_plt, y_min, y_max = [], [], []
    for ep in range(N_ep):
        x_ep_plt.append(ep)
        y_min.append(B_ep[ep][1,0,0])
        print(f"min {B_ep[ep][1,0,0]}, max {B_ep[ep][1,0,n_B-1]}")
        y_max.append(B_ep[ep][1,0,n_B-1]) # B_ep[N_ep] of numpy array [Ns,Nu,Ne]
        for ii in range(n_B):
            x_ensemble.append(ep)
            y_ensemble.append(B_ep[ep][1,0,ii])

    a0 = f2.add_subplot(gs2[0,:])
    a0.scatter(x_ensemble, y_ensemble)
    a0.fill_between(x_ep_plt,y_min,y_max, color='b', alpha=0.1)
    a0.set_title('Values of Bs in the Ensemble at Each Episode')
    a0.set_xlabel('Episode')
    a0.set_ylabel('B value')
    a0.xaxis.set_major_locator(MaxNLocator(integer=True))
    a0.grid()

    # - Plot predicted trajectories for 3 selected episodes:
    # Plot MPC cost and executed trajectory every episode:
    plot_ep = [0, int((N_ep-1)/2), N_ep-1]

    # - Plot evolution of MPC cost:
    y_mpc = [c/mpc_cost_ep[0] for c in mpc_cost_ep]
    b0 = f2.add_subplot(gs2[1,:])
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
        pos_plot[ii].plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2, label='Ground constraint')
        pos_plot[ii].plot(t_eval, x_ep[plot_ep[ii], :, 0], label='z')


        Nt = len(x_th[0])
        ensemble_time_indices = [0,int(Nt/4),int(Nt/3)]
        for index_time_ensemble in ensemble_time_indices:
            for x_th_en in x_th[plot_ep[ii]][index_time_ensemble]: # for every ensemble at time zero 
                pos_plot[ii].plot(
                    t_eval+t_eval[index_time_ensemble], 
                    np.hstack([x_ep[plot_ep[ii]][index_time_ensemble,0],x_th_en[0,:]]), lw=0.5,c='k', alpha=0.5 )

        #b1_lst[ii].fill_between(t_eval, ref[0,:], x_ep[plot_ep[ii], :, 0], alpha=0.2)
        err_norm = (t_eval[-1]-t_eval[0])*np.sum(np.square(x_ep[plot_ep[ii], :, 0].T - ref[0,:]))/x_ep[plot_ep[ii], :, 0].shape[0]
        #b1_lst[ii].text(1.2, 0.5, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))

        pos_plot[ii].set_title('Executed trajectory, ep ' + str(plot_ep[ii]))
        #pos_plot[ii].set_xlabel('Time (sec)')
        pos_plot[ii].set_ylabel('z (m)')
        pos_plot[ii].grid()

        vel_plot.append(f2.add_subplot(gs2[3, ii]))
        vel_plot[ii].plot(t_eval, x_ep[plot_ep[ii], :, 1], label='$\dot{z}$')
        
        for index_time_ensemble in ensemble_time_indices:
            for x_th_en in x_th[plot_ep[ii]][index_time_ensemble]: # for every ensemble at time zero 
                vel_plot[ii].plot(
                    t_eval+t_eval[index_time_ensemble], 
                    np.hstack([x_ep[plot_ep[ii]][index_time_ensemble,1],x_th_en[1,:]]), lw=0.5,c='k', alpha=0.5 )
        vel_plot[ii].set_ylabel('$\dot{z}$ (m/s)')
        vel_plot[ii].grid()


        u_plot.append(f2.add_subplot(gs2[4, ii]))
        u_plot[ii].plot(t_eval[:-1], u_ep[plot_ep[ii], :, 0], label='T')
        u_plot[ii].plot([t_eval[0], t_eval[-2]], [umax+T_hover, umax+T_hover], '--r', lw=2, label='Max thrust')

        for index_time_ensemble in ensemble_time_indices:
            u_plot[ii].plot(
                    t_eval[:-1]+t_eval[index_time_ensemble], T_hover+u_th[plot_ep[ii]][index_time_ensemble][0,:], lw=0.5,c='k', alpha=0.5 )

        u_plot[ii].fill_between(t_eval[:-1], np.zeros_like(u_ep[plot_ep[ii], :, 0]), u_ep[plot_ep[ii], :, 0], alpha=0.2)
        ctrl_norm = (t_eval[-2] - t_eval[0]) * np.sum((np.square(u_ep[plot_ep[ii], :, 0]))/u_ep[plot_ep[ii], :, 0].shape[0])
        u_plot[ii].text(1.2, 11, "$\int u_n^2=${0:.2f}".format(ctrl_norm))
        u_plot[ii].set_title('Executed control effort, ep ' + str(plot_ep[ii]))
        u_plot[ii].set_xlabel('Time (sec)')
        u_plot[ii].set_ylabel('Thrust (N)')
        u_plot[ii].grid()
    pos_plot[-1].legend(loc='lower right')
    u_plot[-1].legend(loc='upper right')

    gs2.tight_layout(f2)
    f2.savefig('core/examples/results/executed_traj.pdf', format='pdf', dpi=2400)
    plt.show()

sp.io.savemat('./core/examples/1d_drone.mat', {'B_ep': B_ep, 
                                                    'N_ep':N_ep, 
                                                    'mpc_cost_ep':mpc_cost_ep,
                                                    't_eval':t_eval, 
                                                    'x_ep':x_ep,
                                                    'u_ep':u_ep,
                                                    'x_th':x_th,
                                                    'u_th':u_th,
                                                    'ground_altitude':ground_altitude,
                                                    'T_hover':T_hover})


plot_summary_EnMPC(B_ep, N_ep, mpc_cost_ep, t_eval, x_ep, u_ep, x_th, u_th, ground_altitude,T_hover)


def plot_interactive_EnMPC(B_ep, N_ep, mpc_cost_ep, t_eval, x_ep):
    f2 = plt.figure(figsize=(12,9))
    gs2 = gridspec.GridSpec(2,1, figure=f2)

    ii = 0
    
    
    b1 = f2.add_subplot(gs2[0, ii])
    b1.plot([t_eval[0], t_eval[-1]], [ground_altitude, ground_altitude], '--r', lw=2, label='Ground constraint')
    b1.plot(t_eval, x_ep[ii, :, 0], label='z')
    b1.fill_between(t_eval, ref[0,:], x_ep[ii, :, 0], alpha=0.2)
    b1.plot(t_eval, x_ep[ii, :, 1], label='$\dot{z}$')
    err_norm = (t_eval[-1]-t_eval[0])*np.sum(np.square(x_ep[ii, :, 0].T - ref[0,:]))/x_ep[ii, :, 0].shape[0]
    b1.text(1.2, 0.5, "$\int (z-z_d)^2=${0:.2f}".format(err_norm))
    b1.set_title('Executed trajectory, ep ' + str(ii))
    b1.set_xlabel('Time (sec)')
    b1.set_ylabel('z, $\dot{z}$ (m, m/s)')
    b1.grid()

    b2 = f2.add_subplot(gs2[1, ii])
    b2.plot(t_eval[:-1], u_ep[ii, :, 0], label='T')
    b2.plot([t_eval[0], t_eval[-2]], [umax+T_hover, umax+T_hover], '--r', lw=2, label='Max thrust')
    b2.fill_between(t_eval[:-1], np.zeros_like(u_ep[ii, :, 0]), u_ep[ii, :, 0], alpha=0.2)
    ctrl_norm = (t_eval[-2] - t_eval[0]) * np.sum((np.square(u_ep[ii, :, 0]))/u_ep[ii, :, 0].shape[0])
    b2.text(1.2, 11, "$\int u_n^2=${0:.2f}".format(ctrl_norm))
    b2.set_title('Executed control effort, ep ' + str(ii))
    b2.set_xlabel('Time (sec)')
    b2.set_ylabel('Thrust (N)')
    b2.grid()
    plt.show()


plot_interactive_EnMPC(B_ep, N_ep, mpc_cost_ep, t_eval, x_ep)


def plot_keedmd_improvement(B_ep, mpc_cost_ep, t_eval, x_ep, u_ep, B_ep_keedmd, mpc_cost_ep_keedmd, t_eval_keedmd, x_ep_keedmd, u_ep_keedmd, ground_altitude, T_hover, N_ep):
    plt.figure(figsize=(6, 4))

    # - Plot evolution of B ensemble:
    n_B = B_ep[0].shape[2]
    x_ensemble, y_ensemble = [], []
    x_ep_plt, y_min, y_max = [], [], []

    x_ensemble_keedmd, y_ensemble_keedmd = [], []
    y_min_keedmd, y_max_keedmd = [], []

    for ep in range(N_ep):
        x_ep_plt.append(ep)
        y_min.append(B_ep[ep][1, 0, 0])
        y_max.append(B_ep[ep][1, 0, n_B - 1])  # B_ep[N_ep] of numpy array [Ns,Nu,Ne]

        y_min_keedmd.append(B_ep_keedmd[ep][1, 0, 0])
        y_max_keedmd.append(B_ep_keedmd[ep][1, 0, n_B - 1])  # B_ep[N_ep] of numpy array [Ns,Nu,Ne]
        for ii in range(n_B):
            x_ensemble.append(ep-0.05)
            y_ensemble.append(B_ep[ep][1, 0, ii])

            x_ensemble_keedmd.append(ep+0.05)
            y_ensemble_keedmd.append(B_ep_keedmd[ep][1, 0, ii])

    plt.subplot(2,1,1)
    plt.scatter(x_ensemble, y_ensemble, color='b', label='State space model')
    plt.fill_between(x_ep_plt, y_min, y_max, color='b', alpha=0.1)
    plt.scatter(x_ensemble_keedmd, y_ensemble_keedmd, color='r', label='Lifted state model')
    plt.fill_between(x_ep_plt, y_min_keedmd, y_max_keedmd, color='r', alpha=0.1)
    plt.title('Values of Bs in the Ensemble at Each Episode')
    plt.xlabel('Episode')
    plt.ylabel('B value')
    plt.xticks([ii for ii in range(N_ep)])
    plt.grid()
    plt.legend(loc='upper right')

    # - Plot evolution of MPC cost:
    y_mpc = [c / mpc_cost_ep[0] for c in mpc_cost_ep]
    y_mpc_keedmd = [c / mpc_cost_ep[0] for c in mpc_cost_ep_keedmd]
    plt.subplot(2,1,2)
    plt.plot(x_ep_plt, y_mpc, label='State space model')
    plt.plot(x_ep_plt, y_mpc_keedmd, label='Lifted state model')
    plt.title('MPC Cost Evolution')
    plt.xlabel('Episode')
    plt.ylabel('Normalized MPC Cost')
    plt.xticks([ii for ii in range(N_ep)])
    plt.grid()
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('core/examples/results/keedmd_comparison.pdf', format='pdf', dpi=2400)
    plt.show()

plot_keedmd_improvement(B_ep, mpc_cost_ep, t_eval, x_ep, u_ep, B_ep_keedmd, mpc_cost_ep_keedmd, t_eval, x_ep_keedmd, u_ep_keedmd, ground_altitude, T_hover, N_ep)