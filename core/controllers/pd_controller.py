from numpy import dot, random

from .controller import Controller

class PDController(Controller):
    """Class for proportional-derivative policies."""

    def __init__(self, pd_dynamics, K_p, K_d, noise_var=0.):
        """Create a PDController object.

        Policy is u = -K_p * e_p - K_d * e_d, where e_p and e_d are propotional
        and derivative components of error.

        Inputs:
        Proportional-derivative dynamics, pd_dynamics: PDDynamics
        Proportional gain matrix, K_p: numpy array
        Derivative gain matrix, K_d: numpy array
        """

        Controller.__init__(self, pd_dynamics)
        self.m = K_p.shape[0]
        self.K_p = K_p
        self.K_d = K_d
        self.noise_var = noise_var
        self.brownian = 0.

    def eval(self, x, t):
        """eval Function to evaluate the controller
        
        Arguments:
            x {numpy array [ns,]} -- state
            t {float} -- time
        
        Returns:
            control action -- numpy array [Nu,]
        """
        e_p = self.dynamics.proportional(x, t)
        e_d = self.dynamics.derivative(x, t)
        self.brownian = self.noise_var*random.randn(self.m)  #TODO: Revert to Brownian or remove additional notation
        return -dot(self.K_p, e_p) - dot(self.K_d, e_d) + self.brownian
