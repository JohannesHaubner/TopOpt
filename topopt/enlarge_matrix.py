from dolfin import *
from dolfin_adjoint import *
import numpy as np
from quadrature.quadrature import get_weights
import scipy.sparse as sps
import scipy.sparse as sparse


def enlarge(A, number):
    size = A.shape[0]
    enlarged_size = A.shape[0]+number
    A.resize((enlarged_size, enlarged_size))
    indices = range(size, enlarged_size)
    val = np.ones(len(indices))
    A = A.tocoo()
    row = np.append(A.row, indices)
    col = np.append(A.col, indices)
    val = np.append(A.data, val)
    A = sps.coo_matrix((val, (row, col)), shape=(enlarged_size, enlarged_size))
    A = A.tocsr()
    return A

if __name__ == "__main__":
    valx = [1, 2, 3]
    indx = [1, 2, 3]
    indy = [2, 1, 3]
    ndofs = 10
    A = sps.csr_matrix((valx, (indx, indy)), shape=(ndofs, ndofs))

    A = enlarge(A, 3)
