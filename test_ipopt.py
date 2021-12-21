import pytest
import preprocessing
import Hs_regularization
import ipopt_solver
import numpy as np

from dolfin import *
from dolfin_adjoint import *

def test_ipopt_cholesky():
    # load Hs matrix
    N = 10
    delta = 1.0
    sigma = 7. / 16
    reg_Hs = Hs_regularization.AssembleHs(N, delta, sigma)
    matrix = reg_Hs.get_matrix(1.0)

    # apply cholesky
    U = ipopt_solver.IPOPTProblem.sparse_cholesky(matrix)

    # check if U^T U = matrix
    UTU = U.transpose().dot(U)
    assert np.all(matrix.todense() - UTU.todense() < 1e-13)

if __name__ == "__main__":
    test_ipopt_cholesky()