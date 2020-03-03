
from numpy.linalg import eigvals
import time

import numpy as np
import scipy as sp
import osqp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .controller import Controller
from ..learning.edmd import Edmd

## Auxiliary functions to support Controllers
def block_diag(M,n):
  """bd creates a sparse block diagonal matrix by repeating M n times
  
  Args:
      M (2d numpy array): matrix to be repeated
      n (float): number of times to repeat
  """
  return sp.sparse.block_diag([M for i in range(n)])

