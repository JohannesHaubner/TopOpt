import pytest
import Hs_regularization
import numpy as np

from dolfin import *
from dolfin_adjoint import *

def test_Hs_matrix():
    N = 15
    delta = 1.0
    sigma = 7./16
    reg_Hs = Hs_regularization.AssembleHs(N, delta, sigma)
    matrix = reg_Hs.get_matrix(1.0)

    # check if matrix is symmetric
    assert np.all(matrix.transpose().todense() - matrix.todense()== 0)
    # check if positive definite
    assert np.all(np.linalg.eigvals(matrix.todense())> 0)

if __name__ == "__main__":
    test_Hs_matrix()