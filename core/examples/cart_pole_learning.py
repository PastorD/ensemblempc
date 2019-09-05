"""Cart Pendulum Example"""

from os import path
import sys
from matplotlib.pyplot import figure, grid, legend, plot, show, subplot, suptitle, title
from numpy import arange, array, concatenate, cos, identity, linspace, ones, sin, tanh, tile, zeros, pi, random, interp, dot, multiply, asarray
#from numpy.random import uniform
from scipy.io import loadmat, savemat
from sys import argv
from core.systems import CartPole
from core.dynamics import LinearSystemDynamics
from core.controllers import PDController, OpenLoopController, MPCController
from core.learning_keedmd import KoopmanEigenfunctions, RBF, Edmd, Keedmd, plot_trajectory

import random
import scipy.sparse as sparse
class CartPoleTrajectory(CartPole):
    def __init__(self, robotic_dynamics, q_d, t_d):
        m_c, m_p, l, g = robotic_dynamics.params
        CartPole.__init__(self, m_c, m_p, l, g)
        self.robotic_dynamics = robotic_dynamics
        self.q_d = q_d
        self.t_d = t_d

    def eval(self, q, t):
        return q - self.desired_state(t)

    def desired_state(self, t):
        return [interp(t, self.t_d.flatten(),self.q_d[ii,:].flatten()) for ii in range(self.q_d.shape[0])]

    def drift(self, q, t):
        return self.robotic_dynamics.drift(q, t)

    def act(self, q, t):
        return self.robotic_dynamics.act(q, t)

#%% ===============================================   SET PARAMETERS    ===============================================

# Define true system
system_true = CartPole(m_c=.5, m_p=.2, l=.4)
n, m = 4, 1  # Number of states and actuators
upper_bounds = array([2.5, pi/3, 2, 2])  # State constraints
lower_bounds = -upper_bounds  # State constraints

# Define nominal model and nominal controller:
A_nom = array([[0., 0., 1., 0.], [0., 0., 0., 1.], [0., -3.924, 0., 0.], [0., 34.335, 0., 0.]])  # Linearization of the true system around the origin
B_nom = array([[0.],[0.],[2.],[-5.]])  # Linearization of the true system around the origin
K_p = -array([[7.3394, 39.0028]])  # Proportional control gains
K_d = -array([[8.0734, 7.4294]])  # Derivative control gains
nominal_sys = LinearSystemDynamics(A=A_nom, B=B_nom)

# Simulation parameters
dt = 1.0e-2  # Time step
N = int(2./dt)  # Number of time steps
t_eval = dt * arange(N + 1) # Simulation time points
noise_var = 0.25  # Exploration noise to perturb controller

# Koopman eigenfunction parameters
eigenfunction_max_power = 3
l1_diffeomorphism = 1e-4
l2_diffeomorphism = 1e0
load_diffeomorphism_model = True
diffeomorphism_model_file = 'diff_model'
diff_n_epochs = 100
diff_train_frac = 0.9
diff_n_hidden_layers = 2
diff_layer_width = 50
diff_batch_size = 64
diff_learn_rate = 1e-2
diff_learn_rate_decay = 0.95

# KEEDMD parameters
l1_keedmd = 1e-2
l2_keedmd = 1e-2

# EDMD parameters
n_lift_edmd = 50
l1_edmd = 1e-2
l2_edmd = 1e-2

#%% ===============================================    COLLECT DATA     ===============================================

# Load trajectories
traj_origin = 'gen_MPC'
if (traj_origin == 'gen_MPC'):
    Ntraj = 10
    t_d = t_eval
    traj_bounds = [1,0.2,1.,1.] # x, theta, x_dot, theta_dot 
    q_d = zeros((n,Ntraj,N+1))
    Q = sparse.diags([0,0,0,0])
    QN = sparse.diags([100000.,100000.,50000.,10000.])
    R = sparse.eye(m)

    mpc_controller = MPCController(affine_dynamics=nominal_sys, 
                                N=200,
                                dt=dt, 
                                umin=array([-10]), 
                                umax=array([10]),
                                xmin=lower_bounds, 
                                xmax=upper_bounds, 
                                Q=Q, 
                                R=R, 
                                QN=QN, 
                                x0=zeros(n), 
                                xr=zeros(n) )
    for ii in range(Ntraj):
        x_0 = asarray([random.uniform(-i,i)  for i in traj_bounds ])
        mpc_controller.eval(x_0,0)
        q_d[:,ii,:] = mpc_controller.parse_result()
    
    plot_traj_gen = True
    if plot_traj_gen:
        figure()
        title('Input Trajectories')
        for j in range(n):
            subplot(n,1,j+1)
            [plot(t_eval, q_d[j,ii,:] , linewidth=2) for ii in range(Ntraj)]
            grid()
        show()
            
        
elif (traj_origin=='load_mat'):
    res = loadmat('./core/examples/cart_pole_d.mat') # Tensor (n, Ntraj, Ntime)
    q_d = res['Q_d']  # Desired states
    t_d = res['t_d']  # Time points
    Ntraj = q_d.shape[1]  # Number of trajectories to execute

    figure()
    title('Input Trajectories')
    for j in range(n):
        subplot(n,1,j+1)
        [plot(t_eval, q_d[j,ii,:] , linewidth=2) for ii in range(Ntraj)]
        grid()
    show()


# Simulate system from each initial condition
outputs = [CartPoleTrajectory(system_true, q_d[:,i,:],t_d) for i in range(Ntraj)]
pd_controllers = [PDController(outputs[i], K_p, K_d, noise_var) for i in range(Ntraj)]
pd_controllers_nom = [PDController(outputs[i], K_p, K_d, 0.) for i in range(Ntraj)]  # Duplicate of controllers with no noise perturbation
xs, us, us_nom, ts = [], [], [], []
for ii in range(Ntraj):
    x_0 = q_d[:,ii,0]
    xs_tmp, us_tmp = system_true.simulate(x_0, pd_controllers[ii], t_eval)
    us_nom_tmp = pd_controllers_nom[ii].eval(xs_tmp.transpose(), t_eval).transpose()
    xs.append(xs_tmp)
    us.append(us_tmp)
    us_nom.append(us_nom_tmp[:us_tmp.shape[0],:])
    ts.append(t_eval)
savemat('./core/examples/results/cart_pendulum_pd_data.mat', {'xs': xs, 't_eval': t_eval, 'us': us, 'us_nom':us_nom})
xs, us, us_nom, ts = array(xs), array(us), array(us_nom), array(ts)

#plot_trajectory(xs[0], q_d[:,0,:].transpose(), us[0], us_nom[0], ts[0])  # Plot simulated trajectory if desired

#%% ===============================================     FIT MODELS      ===============================================

# Construct basis of Koopman eigenfunctions for KEEDMD:
A_cl = A_nom - dot(B_nom,concatenate((K_p, K_d),axis=1))
BK = dot(B_nom,concatenate((K_p, K_d),axis=1))
print('Constructing Koopman eigenfunction basis....')
eigenfunction_basis = KoopmanEigenfunctions(n=n, max_power=eigenfunction_max_power, A_cl=A_cl, BK=BK)
eigenfunction_basis.build_diffeomorphism_model(n_hidden_layers = diff_n_hidden_layers, layer_width=diff_layer_width,
                                               l2=l2_diffeomorphism, batch_size = diff_batch_size)
if load_diffeomorphism_model:
    eigenfunction_basis.load_diffeomorphism_model(diffeomorphism_model_file)
else:
    eigenfunction_basis.fit_diffeomorphism_model(X=xs, t=t_eval, X_d=q_d, learning_rate=diff_learn_rate,
                                                 learning_decay=diff_learn_rate_decay, n_epochs=diff_n_epochs, train_frac=diff_train_frac)
    eigenfunction_basis.save_diffeomorphism_model(diffeomorphism_model_file)
eigenfunction_basis.construct_basis(ub=upper_bounds, lb=lower_bounds)
#eigenfunction_basis.plot_eigenfunction_evolution(xs[-1], t_eval)

# Fit KEEDMD model:
print('Fitting KEEDMD model...')
keedmd_model = Keedmd(eigenfunction_basis, n, l1=l1_keedmd, l2=l2_keedmd)
keedmd_model.fit(xs, us, us_nom, ts)

# Construct basis of RBFs for EDMD:
print('Constructing RBF basis...')
rbf_centers = multiply(random.rand(n_lift_edmd, n),(upper_bounds-lower_bounds))+lower_bounds
rbf_basis = RBF(rbf_centers, n)
rbf_basis.construct_basis()

# Fit EDMD model
print('Fitting EDMD model...')
edmd_model = Edmd(rbf_basis, n, l1=l1_edmd, l2=l2_edmd)
edmd_model.fit(xs, us, us_nom, ts)

#%% ==============================================  EVALUATE PERFORMANCE  =============================================
# Set up trajectory and controller for prediction task:
q_d_pred = q_d[:,4,:]
t_pred = t_d.squeeze()
noise_var_pred = 0.5
output_pred = CartPoleTrajectory(system_true, q_d_pred,t_pred)
pd_controller_pred = PDController(output_pred, K_p, K_d, noise_var_pred)

# Simulate true system (baseline):
x0_pred = q_d_pred[:,0]
xs_pred, us_pred = system_true.simulate(x0_pred, pd_controller_pred, t_pred)

# Create systems for each of the learned models and simulate with open loop control signal us_pred:
#keedmd_sys = LinearSystemDynamics(A=keedmd_model.A, B=keedmd_model.B)
#keedmd_controller = OpenLoopController(keedmd_sys, us_pred, t_pred[:us_pred.shape[0]])
#z0_keedmd = keedmd_model.lift(x0_pred.reshape(x0_pred.shape[0],1), zeros((1,))).squeeze()
#print(z0_keedmd.shape, len(z0_keedmd), keedmd_sys.n)
#zs_keedmd, us_keedmd = keedmd_sys.simulate(z0_keedmd,keedmd_controller,t_pred)
#rint(z0_keedmd.shape, zs_keedmd.shape, us_keedmd.shape)

#edmd_sys = LinearSystemDynamics(A=edmd_model.A, B=edmd_model.B)

#savemat('./core/examples/results/cart_pendulum_pd_data.mat', {'xs': xs, 't_eval': t_eval, 'us': us, 'us_nom':us_nom})
#xs, us, us_nom, ts = array(xs), array(us), array(us_nom), array(ts)