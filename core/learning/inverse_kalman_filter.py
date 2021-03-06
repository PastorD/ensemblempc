from .utils import differentiate_vec
from sklearn import linear_model
from scipy.linalg import expm
from numpy import array, concatenate, zeros, dot, linalg, eye, ones, std, where, divide, multiply, tile, argwhere, diag, copy, ones_like
from .basis_functions import BasisFunctions
from .learner import Learner
from .eki import EKS
import numpy as np
import scipy
import scipy.signal as signal
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt


def hp(x):
    # use to to plot a numpy array
    import matplotlib.pyplot as plt
    plt.matshow(x)
    plt.colorbar()
    plt.show()

class InverseKalmanFilter(Learner):
    '''
    Transforms a parametrized dynamics problem as a Inverse Kalman Inversion problem
    '''
    def __init__(self, A, TrueB, E, eta_0, B_ensemble, dt, nk, maxiter=15):

        self.A = A
        self.B_ensemble = B_ensemble
        self.Ns = self.B_ensemble.shape[0]
        self.Nu = self.B_ensemble.shape[1]
        self.Ne = self.B_ensemble.shape[2]
        B_ensemble_flat = np.reshape(B_ensemble, (self.Nu*self.Ns,self.Ne))
        G = lambda theta,y: 0

        self.eks = EKS(B_ensemble_flat, G, eta_0, 
              true_theta=TrueB.flatten(), maxiter=maxiter, max_error= 1e-6)
        self.Bshape = TrueB.shape
        self.dt = dt
        self.nk = nk    
        self.get_multistep_matrices(TrueB)
        self.E = E
    
    def get_multistep_matrices(self,B):
        # #! Prep matrices for prediction
        # build A^{nk}
        lin_model_d = signal.cont2discrete((self.A,B,np.identity(self.Ns),zeros((self.Ns,1))),self.dt)
        Ad = lin_model_d[0]
        Bd = lin_model_d[1]
        xpm = scipy.linalg.expm(self.A*self.dt*self.nk)
        
        # # build ABM as in x(k)=Ad^k+ABM @ uvector
        self.ABM = Bd 
        self.An  = Ad
        for i in range(self.nk-1):
            self.ABM = np.hstack([self.An @ Bd,self.ABM])
            self.An = self.An @ Ad

        # Test Prep Matrices
        check_ab = False
        if check_ab:
            x0  = np.random.rand(self.Ns)
            xd = x0.copy()
            xc = x0.copy()

            # Store data Init
            xst = np.zeros((self.Ns,self.nk))
            ust = np.zeros((self.Nu,self.nk))

            # Simulate in closed loop
            for i in range(self.nk):
                # Fake pd controller
                ctrl = np.zeros(self.Nu,) 
                ctrl = np.random.rand(self.Nu,)
                xd = Ad @ xd + Bd @ ctrl
                xc = solve_ivp(lambda t,x: self.A @ x + B @ ctrl, [0, self.dt], xc, atol=1e-6, rtol=1e-6).y[:, -1] 
         
                # Store Data
                xst[:,i] = xd
                ust[:,i] = ctrl

            #xc2 = solve_ivp(lambda t,x: self.A @ x + B @ ust[:,np.max([np.int(t/self.dt),self.nk-1])], [0, self.dt*self.nk], x0, atol=1e-6, rtol=1e-6).y[:, -1] 
            #print(f"cont 2{xc2}")
            x_multistep = self.An@x0 + self.ABM@ust.flatten()
            print(f"multistep {x_multistep}")
            print(f"discrete {xd}")
            print(f"continous {xc}")
            print(f"ctrl")
        

    def fit(self, X, U):
        """
        Fit a learner

        Inputs:
        - X: state with all trajectories, list [Ntraj] numpy array [ns,Nt]
        - X_dot: time derivative of the state
        - U: control input, numpy 3d array [NtrajxN, nu]
        - t: time, numpy 2d array [Ntraj, N]
        """
        Ntraj = len(X)
        
        debug = False
        if debug:
            plt.figure()
            plt.subplot(2,1,1,xlabel="time", ylabel="X")
            for Xtraj in X:
                for i in range(self.Ns):
                    plt.plot(Xtraj[i,:], linewidth=1,label=f'state {i}') 
            plt.grid()
            plt.title("State")
            plt.legend()
            plt.subplot(2,1,2,xlabel="U", ylabel=f"U")
            for Utraj in U:
                for i in range(self.Nu):
                    plt.plot(Utraj[i,:], linewidth=1,label=f'Input {i}') 
            plt.grid()
            plt.title("Input")
            plt.legend()
            plt.show()
            plt.savefig(f"fit_debug_states.pdf", format='pdf', dpi=1200,bbox_inches='tight')




        
        shrink_debug = False
        if (shrink_debug):
            shrink_rate = 0.5
            B_mean = np.mean(self.B_ensemble,axis=2)
            self.new_ensamble = self.B_ensemble
            for i in range(self.Ne):
                self.new_ensamble[:,:,i] = B_mean + shrink_rate*(self.B_ensemble[:,:,i]-B_mean)
        else:

            Nt = X[0].shape[1] # number of 
            Ntk = Nt - self.nk # number of columns per trajectory 
            Ng = Ntk*Ntraj # number of columns of G
            Ngs = Ng*self.Ns # total size of G flatten
            Ym = np.empty((Ntraj,self.Ns,Ntk))
            for i_traj, Xtraj in enumerate(X):
                Ydiff = Xtraj[:,self.nk:] - Xtraj[:,:-self.nk]
                Ym[i_traj,:,:] = Ydiff
            Ym_flat = Ym.flatten()
            self.eks.G = lambda Bflat: self.Gdynamics(Bflat,X,U)
            self.B_ensemble_flat =  self.B_ensemble.reshape(-1, self.B_ensemble.shape[-1]) # [NsNu,Ne]
            print(f"new {self.B_ensemble_flat}")
            self.new_ensemble_flat = self.eks.solveIP(self.B_ensemble_flat, Ym_flat)
            print(f"new {self.B_ensemble_flat}")
            self.new_ensamble = self.new_ensemble_flat.reshape((self.Ns,self.Nu,self.Ne))
    
        self.B_ensemble = self.new_ensamble.copy()

    def Gdynamics(self,Bflat, X, U):
        """
        Create G in EKI y = G(theta)
        
        Ng: number of measurements
        
        Arguments:
            Bflat {numpy array [Ns Nu]} -- flat dynamical parameters
            X {numpy array [Ntraj][Ns,Nt]} -- data
            U {numpy array [Ntraj[Nu,Nt]]} -- input
        
        Returns:
            numpy array [Ng,] -- G(theta)
        """
        Ntraj = len(X)
        Nt = X[0].shape[1] # number of 
        Ntk = Nt - self.nk # number of columns per trajectory 
        Ng = Ntk*Ntraj # number of columns of G
        Ngs = Ng*self.Ns # total size of G flatten
        G = np.empty((Ntraj,self.Ns,Ntk))

        B = Bflat.reshape(self.Bshape)
        #self.get_multistep_matrices(B)
        
        for i_traj, (Xtraj, Utraj) in enumerate(zip(X,U)):
            for i in range(Ntk):
                xc = Xtraj[:,i] # init of nk steps
                for multistep_index in range(self.nk):
                    ctrl = Utraj[:,i+multistep_index]
                    xc = solve_ivp(lambda t,x: self.A @ x + B @ ctrl + self.E, [0, self.dt], xc, atol=1e-6, rtol=1e-6).y[:, -1] 
                Gi = xc-Xtraj[:,i]
                G[i_traj,:,i] = Gi

                #ctrl = U[:,i:i+self.nk]
                #f_x_dot = lambda t,x: self.A @ x + B @ ctrl[int(t/dt)]
                #Xplus = solve_ivp(f_x_dot, [0, dt*nk], X[:,j], atol=1e-6, rtol=1e-6).y[:, -1] 
                #G[:,i] = xc-X[:,i]
                #G[:,i] = self.An @ X[:,i] + self.ABM @ U[:,i:i+self.nk].flatten()#-X[:,i]
        return G.flatten()
        

    def predict(self,X, U):
        pass
