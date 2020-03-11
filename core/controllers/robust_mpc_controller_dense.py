

from numpy import zeros
from numpy.linalg import eigvals
import time

import numpy as np
import scipy as sp
import scipy.io as sio
import scipy.sparse as sparse
import osqp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .controller import Controller
from ..learning.edmd import Edmd
from .controller_aux import block_diag, build_boldAB


class RobustMpcDense(Controller):
    """
    Class for controllers MPC.

    MPC are solved using osqp.

    Use lifting=True to solve MPC in the lifted space
    """
    def __init__(self, linear_dynamics, N, dt, 
                umin, umax, xmin, xmax, 
                Q, R, QN, xr, 
                plotMPC_filename=None,
                edmd_object=None, 
                name="noname", 
                D=None,
                ensemble=None,
                gather_thoughts=True):
        """__init__ [summary]
        
        osqp state: 
        hard constraints [state row [Ns Nt], control [Nu Nt]]
        soft constraints [state row [Ns Nt], control [Nu Nt], slack [2Ns Nt]]

        Arguments:
            linear_dynamics {dynamical sytem} -- it contains the A and B matrices in continous time
            N {integer} -- number of timesteps
            dt {float} -- time step in seconds
            umin {numpy array [Nu,]} -- minimum control bound
            umax {numpy array [Nu,]} -- maximum control bound
            xmin {numpy array [Ns,]} -- minimum state bound
            xmax {numpy array [Ns,]} -- maximum state bound
            Q {numpy array [Ns,Ns]} -- state cost matrix
            R {numpy array [Nu,Nu]} -- control cost matrix
            QN {numpy array [Ns,]} -- final state cost
            xr {numpy array [Ns,]} -- reference trajectory
        
        Keyword Arguments:
            plotMPC {bool} -- flag to plot results (default: {False})
            plotMPC_filename {str} -- plotting filename (default: {""})
            lifting {bool} -- flag to use state lifting (default: {False})
            edmd_object {edmd object} -- lifting object. It contains projection matrix and lifting function (default: {Edmd()})
            name {str} -- name for all saved files (default: {"noname"})
            soft {bool} -- flag to enable soft constraints (default: {False})
            D {numpy array []} -- cost matrix for the soft variables (default: {None})
        """

        Controller.__init__(self, linear_dynamics)

        # Load arguments
        Ac, Bc = linear_dynamics.linear_system()
        [nx, nu] = Bc.shape
        ns = xr.shape[0]

        if edmd_object is not None:
            self.C = edmd_object.C
            self.edmd_object = edmd_object
            self.lifting = True
        else:
            self.lifting = False
            self.C = sparse.eye(ns)
            
        if plotMPC_filename is not None:
            self.plotMPC = True
            self.plotMPC_filename = plotMPC_filename
        else:
            self.plotMPC = False

        if D is not None:
            self.soft = True
            self.D = D
        else:
            self.soft = False

        self.dt = dt
        self.q_d = xr

        self.Q = Q
        self.R = R

        self.nu = nu
        self.nx = nx
        self.ns = ns
        
        

        # Total desired path
        if self.q_d.ndim==2:
            self.Nqd = self.q_d.shape[1]
            xr = self.q_d[:,:N]

        # Prediction horizon
        self.N = N
        x0 = np.zeros(nx)
        self.run_time = np.zeros([0,])

        # thoughts
        self.gather_thoughts = gather_thoughts
        if self.gather_thoughts: 
            self.xe_th = []
            self.u_th = []
               

        # Check Xmin and Xmax
        if  xmin.shape[0]==ns and xmin.ndim==1: # it is a single vector we tile it
            x_min_flat = np.kron(np.ones(N), xmin)
            x_max_flat = np.kron(np.ones(N), xmax)
        elif xmin.shape[0]==ns*N: # if it is a long vector it is ok
            x_min_flat = xmin
            x_max_flat = xmax
        elif xmin.shape[0] == ns and xmin.shape[1] == N: # if it is a block we flatten it
            x_min_flat = np.reshape(xmin,(N*ns,),order='F')
            x_max_flat = np.reshape(xmax,(N*ns,),order='F')
        else:
            raise ValueError('xmin has wrong dimensions. xmin shape={}'.format(xmin.shape))
        self.x_min_flat = x_min_flat 
        self.x_max_flat = x_max_flat


        # Check Umin and Umax
        if  umin.shape[0]==nu and umin.ndim==1:
            u_min_flat = np.kron(np.ones(N), umin)
            u_max_flat = np.kron(np.ones(N), umax)
        elif umin.shape[0]==nu*N:
            u_min_flat = umin
            u_max_flat = umax
        elif umin.shape[0] == nu and umin.shape[1] == N: 
            u_min_flat = np.reshape(umin,(N*nu,),order='F')
            u_max_flat = np.reshape(umax,(N*nu,),order='F')
        else:
            raise ValueError('umin has wrong dimensions. Umin shape={}'.format(umin.shape))
        self.u_min_flat = u_min_flat 
        self.u_max_flat = u_max_flat 

        self.ensemble = ensemble
        if ensemble is not None:
            self.Ne = ensemble.shape[2]
            self.bA_e = []
            self.bB_e = []
            for i in range(self.Ne):
                lin_model_e = sp.signal.cont2discrete((Ac,ensemble[:,:,i],self.C,zeros((ns,1))),dt)
                Ad_e = sparse.csc_matrix(lin_model_e[0]) 
                Bd_e = sparse.csc_matrix(lin_model_e[1]) 
                a_temp, B_temp = build_boldAB( Ad_e, Bd_e, N)
                self.bA_e.append(a_temp)
                self.bB_e.append(B_temp)

        lin_model_d = sp.signal.cont2discrete((Ac,Bc,self.C,zeros((ns,1))),dt)
        a,B = build_boldAB( sparse.csc_matrix(lin_model_d[0]), sparse.csc_matrix(lin_model_d[1]) , N)
        self.a = a
        self.B = B

        # Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1)) 
        Rbd = sparse.kron(sparse.eye(N), R)
        Qbd = sparse.kron(sparse.eye(N), Q)

        # Compute Block Diagonal elements
        self.Cbd = sparse.kron(sparse.eye(N), self.C)
        CQCbd  = self.Cbd.T @ Qbd @ self.Cbd
        self.CtQ = self.C.T @ Q
        Cbd = self.Cbd
        
        # Compute quadratic cost matrix
        P = Rbd + B.T @ CQCbd @ B            

        # Compute linear cost vector
        self.BTQbda =  B.T @ CQCbd @ a            
        Aineq_u = sparse.eye(N*nu)
        x0aQb = self.BTQbda @ x0
        xrQB  = B.T @ np.reshape(self.CtQ.dot(xr),(N*nx,),order='F')
        q = x0aQb - xrQB 

        # Compute inequalities
        if ensemble is not None:
            l = u_min_flat
            u = u_max_flat
            for i in range(self.Ne ):
                l = np.hstack([x_min_flat - Cbd @ self.bA_e[i] @ x0,l])
                u = np.hstack([x_max_flat - Cbd @ self.bA_e[i] @ x0,u])

            A = Aineq_u.tocsc()
            for i in range(self.Ne ):
                A = sparse.vstack([Cbd @ self.bB_e[i],A]).tocsc()
        else:
            l = np.hstack([x_min_flat - Cbd @ a @ x0, u_min_flat])
            u = np.hstack([x_max_flat - Cbd @ a @ x0, u_max_flat])

            Aineq_x = Cbd @ B
            A = sparse.vstack([Aineq_x, Aineq_u]).tocsc()

        if self.soft:
            if ensemble is not None:
                Nineq = N*self.Ne
            else:
                Nineq = N
            self.Nineq = Nineq
            Pdelta = sparse.kron(sparse.eye(Nineq), D)
            qdelta = np.zeros(Nineq*ns)
            Adelta = sparse.csc_matrix(np.vstack([np.eye(Nineq*ns),np.zeros((N*nu,Nineq*ns))]))
            A = sparse.hstack([A, Adelta])
            q = np.hstack([q,qdelta])
            P = sparse.block_diag([P,Pdelta])

        plot_matrices = False
        if plot_matrices:
            #! Visualize Matrices
            fig = plt.figure()

            fig.suptitle("QP Matrices to solve MP in dense form. N={}, nx={}, nu={}".format(N,nx,nu),fontsize=20)
            plt.subplot(2,4,1,xlabel="Ns*(N+1)", ylabel="Ns*(N+1)")
            plt.imshow(a.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("a in $x=ax_0+bu$")
            plt.subplot(2,4,2,xlabel="Ns*(N+1)", ylabel="Nu*N")
            plt.imshow(B.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("b in $x=ax_0+bu$")
            plt.subplot(2,4,3,xlabel="ns*(N+1) + ns*(N+1) + nu*N", ylabel="Ns*(N+1)+Nu*N")
            plt.imshow(A.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("A total in $l\\leq Ax \\geq u$")
            plt.subplot(2,4,4)
            plt.imshow(P.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("P in $J=u^TPu+q^Tu$")
            plt.subplot(2,4,5)
            plt.imshow(Qbd.toarray(),  interpolation='nearest', cmap=cm.Greys_r)
            plt.title("Qbd")


            #! Visualize Vectors
            plt.subplot(2,4,6)
            plt.plot(l)
            plt.title('l in  $l\\leq Ax \\geq u$')
            plt.grid()
            plt.subplot(2,4,7)
            plt.plot(u)
            plt.title("l in  $l\\leq Ax \\geq u$")
            plt.grid()
            plt.subplot(2,4,8)
            plt.plot(q)
            plt.title("q in $J=u^TPu+q^Tu$")
            plt.grid()
            plt.tight_layout()
            #plt.savefig("MPC_matrices_for_"+name+".pdf",bbox_inches='tight',format='pdf', dpi=2400)
            #plt.close()
            plt.show()


            if ensemble is not None:
                fig = plt.figure()
                for i in range(self.Ne):
                    plt.subplot(self.Ne,1,i+1)
                    plt.imshow(self.bB_e[i].toarray(),  interpolation='nearest', cmap=cm.Greys_r)
                    plt.title(f"b in $x=ax_0+bu$ b={ensemble[1,0,i]}")


                plt.show()


        # Create an OSQP object
        self.prob = osqp.OSQP()

        # Setup workspace
        self.prob.setup(P=P.tocsc(), q=q, A=A, l=l, u=u, warm_start=True, verbose=False)

        if self.plotMPC:
            # Figure to plot MPC thoughts
            self.fig, self.axs = plt.subplots(self.ns+self.nu)
            if nx==4:
                ylabels = ['$x$', '$\\theta$', '$\\dot{x}$', '$\\dot{\\theta}$']
            else:
                ylabels = [f"state {i}" for i in range(nx)]

            for ii in range(self.ns):
                self.axs[ii].set(xlabel='Time(s)',ylabel=ylabels[ii])
                self.axs[ii].grid()
            for ii in range(self.ns,self.ns+self.nu):
                self.axs[ii].set(xlabel='Time(s)',ylabel='u')
                self.axs[ii].grid()

    def eval(self, x, t):
        '''
        Args:
        - x, numpy 1d array [ns,]
        - time, t, float
        '''
        time_eval0 = time.time()
        N, ns, nu, nx = [self.N, self.ns, self.nu, self.nx]
        self.x0 = x

        tindex = int(np.ceil(t/self.dt)) 
            
        # Update the local reference trajectory
        if (tindex+N) < self.Nqd: # if we haven't reach the end of q_d yet
            xr = self.q_d[:,tindex:tindex+N]
        else: # we fill xr with copies of the last q_d
            xr = np.hstack( [self.q_d[:,tindex:],np.transpose(np.tile(self.q_d[:,-1],(N-self.Nqd+tindex,1)))])

        # Construct the new _osqp_q objects
        if (self.lifting):
            x = np.transpose(self.edmd_object.lift(x.reshape((x.shape[0],1)),xr[:,0].reshape((xr.shape[0],1))))[:,0]

            BQxr  = self.B.T @ np.reshape(self.CtQ.dot(xr),(N*nx,),order='F')
        else:
            BQxr  = self.B.T @ np.reshape(self.Q.dot(xr),(N*nx,),order='F')


        if self.ensemble is not None:
            l = self.u_min_flat
            u = self.u_max_flat
            for i in range(self.Ne):
                l = np.hstack([self.x_min_flat - self.Cbd @ self.bA_e[i] @ x,l])
                u = np.hstack([self.x_max_flat - self.Cbd @ self.bA_e[i] @ x,u])
        else:
            l = np.hstack([self.x_min_flat - self.Cbd @ self.a @ x, self.u_min_flat])
            u = np.hstack([self.x_max_flat - self.Cbd @ self.a @ x, self.u_max_flat])


        # Update initial state
        BQax0 = self.BTQbda @ x
        q = BQax0  - BQxr

        if self.soft:
            qdelta = np.zeros(self.Nineq*ns)
            q = np.hstack([q,qdelta])        

        self.prob.update(q=q,l=l,u=u)

        #print('Time Setup {:.2f}ms'.format(1000*(time.time()-time_eval0)))
        time_eval0 = time.time() 
        ## Solve MPC Instance
        self._osqp_result = self.prob.solve()

        #print('Time Solve {:.2f}ms'.format(1000*(time.time()-time_eval0)))
        time_eval0 = time.time() 

        # Check solver status
        if self._osqp_result.info.status != 'solved':
            print(f'ERROR: MPC DENSE coudl not be solved at t ={t}, x = {x}')
            raise ValueError('OSQP did not solve the problem!')

        if self.plotMPC:
            self.plot_MPC(t, x, xr, tindex)

        self.run_time = np.append(self.run_time,self._osqp_result.info.run_time)
        self.uoutput = self._osqp_result.x[:nu]

        if self.gather_thoughts:            
            self.xe_th.append(self.get_ensemble_state_prediction())
            self.u_th.append(self.get_control_prediction())

        return  self.uoutput

    def get_state_prediction(self):
        u_flat = self._osqp_result.x[:self.nu*self.N]
        return np.reshape(self.a @ self.x0 + self.B @ u_flat,(self.N,self.nx)).T 

    def get_ensemble_state_prediction(self):
        u_flat = self._osqp_result.x[:self.nu*self.N]
        z_b = [np.reshape(self.bA_e[i] @ self.x0 + self.bB_e[i] @ u_flat,(self.N,self.nx)).T  for i in range(self.Ne)]
        return z_b
 

    def use_u(self,x,uraw):
        """parse_result obtain state from MPC optimization
        
        Arguments:
            x {numpy array [Ns,]} -- initial state
            u {numpy array [Nu*N]} -- control action
        
        Returns:
            numpy array [Ns,N] -- state in the MPC optimization
        """
        u = uraw.reshape(self.N*self.nu)
        return  np.transpose(np.reshape( self.a @ x + self.B @ u, (self.N,self.nx)))

    def get_thoughts_traj(self):
        return self.xe_th, self.u_th

    def get_control_prediction(self):
        """get_control_prediction parse control command from MPC optimization
        
        Returns:
            numpy array [N,Nu] -- control command along MPC optimization
        """
        return np.transpose(np.reshape( self._osqp_result.x[-self.N*self.nu:], (self.N,self.nu)))

    def plot_MPC(self, current_time, x0, xr, tindex):
        """plot_MPC Plot MPC thoughts
        
        Arguments:
            current_time {float} -- current time
            x0 {numpy array [Ns,]} -- current state
            xr {numpy array [Ns,N]} -- reference state
            tindex {float} -- time index along reference trajectory
        """

        #* Unpack OSQP results
        N, ns, nu = [self.N, self.ns, self.nu]

        u_flat = self._osqp_result.x
        osqp_sim_state =  np.reshape(self.a @ x0 + self.B @ u_flat,(N,nx)).T
        osqp_sim_forces = np.reshape(u_flat,(N,nu)).T

        if self.lifting:
            osqp_sim_state = np.dot(self.C,osqp_sim_state)

        pos = current_time/(self.Nqd*self.dt) # position along the trajectory
        time = np.linspace(current_time,current_time+N*self.dt,num=N)

        
        for ii in range(self.ns):
            if (tindex==0):
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos],label='x_0')
            elif (tindex==self.Nqd-2):
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos],label='x_f')
            else:
                self.axs[ii].plot(time,osqp_sim_state[ii,:],color=[0,1-pos,pos])
        for ii in range(self.nu):
            self.axs[ii+self.ns].plot(time,osqp_sim_forces[ii,:],color=[0,1-pos,pos])

    def update(self, xmin=None, xmax=None, umax=None, umin= None, Q=None):
        """update Change QP parameters
        
        Keyword Arguments:
            xmin {numpy array [Ns,]} -- minimum state bound (default: {None})
            xmax {numpy array [Ns,]} -- maximum state bound (default: {None})
            umax {numpy array [Nu,]} -- maximum command bound (default: {None})
            umin {numpy array [Nu,]} -- minimum command bound (default: {None})
            Q {numpy array [Ns,Ns]} -- state cost matrix (default: {None})
        
        """
        
        N, ns, nu = [self.N, self.ns, self.nu]
        if xmin is not None and xmax is not None:
            # Check Xmin and Xmax
            if  xmin.shape[0]==ns and xmin.ndim==1: # it is a single vector we tile it
                x_min_flat = np.kron(np.ones(N), xmin)
                x_max_flat = np.kron(np.ones(N), xmax)
            elif xmin.shape[0]==ns*N and xmin.ndim==1: # if it is a long vector it is ok
                x_min_flat = xmin
                x_max_flat = xmax
            elif xmin.shape[0] == ns and xmin.shape[1] == N: # if it is a block we flatten it
                x_min_flat = np.reshape(xmin,(N*ns,),order='F')
                x_max_flat = np.reshape(xmax,(N*ns,),order='F')
            else:
                raise ValueError('xmin has wrong dimensions. xmin shape={}'.format(xmin.shape))
            self.x_min_flat = x_min_flat 
            self.x_max_flat = x_max_flat

        if umin is not None and umax is not None: #TODO check it works 
        # Check Umin and Umax
            if  umin.shape[0]==nu and umin.ndim==1:
                u_min_flat = np.kron(np.ones(N), umin)
                u_max_flat = np.kron(np.ones(N), umax)
            elif umin.shape[0]==nu*N and umin.ndim==1:
                u_min_flat = umin
                u_max_flat = umax
            elif umin.shape[0] == nu and umin.shape[1] == N: 
                u_min_flat = np.reshape(umin,(N*nu,),order='F')
                u_max_flat = np.reshape(umax,(N*nu,),order='F')
            else:
                raise ValueError('umin has wrong dimensions. Umin shape={}'.format(umin.shape))
            self.u_min_flat = u_min_flat 
            self.u_max_flat = u_max_flat 

        if Q is not None:
            raise ValueError('Q changes is not implemented') #TODO implemented Q change

            """             a, B = [self.a, self.B]
            Qbd = sparse.kron(sparse.eye(N), Q)

            P = Rbd + B.T @ Qbd @ B
            self.BTQbda =  B.T @ Qbd @ a
            self.prob.update(P=P,l=l,u=u) """
            
    def finish_plot(self, x, u, u_pd, time_vector, filename):
        """
        Call this function to plot extra lines.

        - x: state, numpy 2darray [Nqd,n] 
        - u, input from this controller [Nqd-1,n] 
        - u_pd, input from a PD controller [Nqd-1,n] 
        - time_vector, 1d array [Nqd
        - filename, string
        """
        u = u.squeeze()
        u_pd = u_pd.squeeze()
        
        self.fig.suptitle(filename[:-4], fontsize=16)
        for ii in range(self.ns):
            self.axs[ii].plot(time_vector, self.q_d[ii,:], linewidth=2, label='$x_d$', color=[1,0,0])
            self.axs[ii].plot(time_vector, x[ii,:], linewidth=2, label='$x$', color=[0,0,0])
            self.axs[ii].legend(fontsize=10, loc='best')
        self.fig.savefig(filename,format='pdf', dpi=2400)
        plt.close()



 