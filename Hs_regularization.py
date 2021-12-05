from dolfin import *
from dolfin_adjoint import *
import numpy as np
import scipy.sparse as sps
from scipy.linalg import fractional_matrix_power, solve

class Regularization:
    def __init__(self, N, delta, weighting):
        """
        we assume the mesh size for the degrees of freedom to be uniform quadrilateral in x and y direction
        with numbering
        |                   |                   |
        ____________________________________________
        |                   |                   |

        |    nx  + 0        |     nx  +1        |

        |                   |                   |

        |                   |                   |
        ____________________________________________
        |                   |                   |

        |         0         |         1         |

        |                   |                   |

        |                   |                   |
        ____________________________________________

        as matrix we consider L2-massmatrix + weighting * Hs-matrix
        """

        # entries of local stencil
        self.sigma = 7./16
        self.h = 1.0/N
        self.nx = int(delta*N)
        self.ny = int(N)
        self.loc = self.__get_entries_of_local_stencils()
        Hs_glob_matrix = self.__get_global_matrix()
        L2_glob_matrix = self.__get_L2_matrix()
        self.glob_matrix = weighting * Hs_glob_matrix + L2_glob_matrix

        # the following step is expensive and not doable in 3d, just needed for IPOPT, see discussion Sec. 7.7
        self.glob_matrix_m05 = self.__inverse_of_square_root(self.glob_matrix)

    def initial_point_trafo(self, x):
        """
        solve the system (H^s matrix)^(-0.5) * y = x
        """
        return solve(self.glob_matrix_m05, x, assume_a = 'pos')

    def get_matrix(self):
        """
        returns the H^s matrix
        """
        return self.glob_matrix

    def get_transformation_matrix(self):
        """
        return the inverse of the square root of the H^s matrix
        """
        return self.glob_matrix_m05

    def transform(self, x):
        """
        x --> (H^s matrix)^(-1/2)*x, needed for "discrete hack" fo perform IPOPT with the correct inner product
        (see Sec. 7.7)
        """
        return np.dot(self.glob_matrix_m05,x)

    def transform_chainrule(self, djy):
        return np.dot(self.glob_matrix_m05.T, djy)

    def __get_entries_of_local_stencils(self):
        """
        The values were computed using quadrature rules based on the integrals presented in Sec.7 of the manuscript
        """
        h = self.h
        sigma = self.sigma
        prefac = -2*h**(2.-2*sigma)

        loc = {}

        loc['e2'] = prefac*(2./(1. - 2.*sigma) *1.039906 + 2.0/(2 - 2*sigma)*(-0.695978))
        loc['e3'] = prefac*(4./(2. - 2.*sigma) *0.210645)
        loc['e4'] = prefac*1.6422e-1
        loc['e5'] = prefac*1.1512e-1
        loc['e6'] = prefac*4.8272e-2
        loc['e7'] = prefac*3.5498e-2
        loc['e8'] = prefac*2.5427e-2
        loc['e9'] = prefac*6.9627e-3
        loc['e10'] = prefac*2.1142e-4
        loc['e11'] = prefac*9.2385e-4
        loc['e12'] = prefac*3.7609e-4
        loc['e13'] = prefac*1.1380e-5

        loc['e1a'] = (-4 * loc['e2'] - 4 * loc['e3'] - 4 * loc['e4'] - 8 * loc['e5'] - 4 * loc['e6'] - 4 * loc['e7']
                      - 8 * loc['e8'] - 8 * loc['e9'] - 4 * loc['e10'] - 4 * loc['e11'] - 8 * loc['e12'] - 8 * loc[
                          'e13'])

        # formulas in Figure 7.1:
        loc['e1b'] = (loc['e1a'] + loc['e11'] + 2*loc['e12'] + 2*loc['e13'])
        loc['e1c'] = (loc['e1a'] + 2*loc['e11'] + 4*loc['e12'] + 4*loc['e13'])
        loc['e1d'] = (loc['e1b'] + loc['e7'] + 2*loc['e8']  +2*loc['e9'] + 2*loc['e10'])
        loc['e1e'] = (loc['e1d'] + loc['e11'] + 2*loc['e12'] + 2*loc['e13'])
        loc['e1f'] = (loc['e1c'] + 2*loc['e7'] +4*loc['e8'] + 4*loc['e9'] + 3*loc['e10'])
        loc['e1g'] = (loc['e1d'] + loc['e4'] + 2*loc['e5'] + 2*loc['e6'] + 2*loc['e9'] + 2*loc['e13'])
        loc['e1h'] = (loc['e1g'] + loc['e11'] + 2* loc['e12'] + loc['e13'])
        loc['e1i'] = (loc['e1h'] + loc['e7'] + 2*loc['e8'] + loc['e9'] + loc['e10'])
        loc['e1j'] = (loc['e1i'] + loc['e4'] + 2*loc['e5'] + loc['e6'] + loc['e9'] + loc['e13'])
        loc['e1k'] = (loc['e1g'] + loc['e2'] + 2*loc['e3'] +2* loc['e5'] + 2*loc['e8'] + 2*loc['e12'])
        loc['e1l'] = (loc['e1k'] + loc['e11'] + loc['e12'] + loc['e13'])
        loc['e1m'] = (loc['e1l'] + loc['e7'] + loc['e8'] + loc['e9'] + loc['e10'])
        loc['e1n'] = (loc['e1m'] + loc['e4'] + loc['e5'] + loc['e6'] + loc['e9'] + loc['e13'])
        loc['e1o'] = (loc['e1n'] + loc['e2'] + loc['e3'] + loc['e5'] + loc['e8'] + loc['e12'])

        return loc

    def __get_L2_matrix(self):
        """
        compute L2-matrix for DG0-function on uniform quadrilateral mesh with mesh-width self.h
        """
        ndofs = self.nx*self.ny
        ind = np.asarray(range(ndofs))
        val = self.h*self.h*np.ones(ndofs)
        return sps.csr_matrix((val, (ind, ind)), shape=(ndofs, ndofs))

    def __get_global_matrix(self):
        """
        numerically approximate the H^s-matrix, as described in Sec. 7
        """

        # values in local matrix
        loc = self.loc
        ent = [loc['e1a'], loc['e2'], loc['e3'], loc['e4'], loc['e5'], loc['e6'], loc['e7'], loc['e8'],
               loc['e9'], loc['e10'], loc['e11'], loc['e12'], loc['e13']]
        rep = [1, 4, 4, 4, 8, 4, 4, 8, 8, 4, 4, 8, 8]
        valloc = np.repeat(np.array(ent), np.array(rep))

        # indexshift
        nx = self.nx
        nxh = self.nx + 4 #add 4 artificial columns to be able to work with indexshifts
        ny = self.ny
        ndofsh = int(nxh*ny)
        ndofs = nx*ny

        indexshift = [0,
                      1, -1, nxh, -nxh,
                      nxh-1, nxh+1, -nxh-1, -nxh+1,
                      2, -2, 2*nxh, -2*nxh,
                      nxh-2, nxh+2, -nxh-2, -nxh+2, 2*nxh+1, 2*nxh-1, -2*nxh+1, -2*nxh-1,
                      2*nxh+2, 2*nxh-2, -2*nxh+2, -2*nxh-2,
                      3, -3, 3*nxh, -3*nxh,
                      nxh+3, nxh-3, -nxh-3, -nxh+3, 3*nxh+1, 3*nxh-1, -3*nxh-1, -3*nxh+1,
                      2*nxh+3, 2*nxh-3, -2*nxh+3, -2*nxh-3, 3*nxh+2, 3*nxh-2, -3*nxh-2, -3*nxh+2,
                      3*nxh+3, 3*nxh-3, -3*nxh-3, -3*nxh+3,
                      4, -4, 4*nxh, -4*nxh,
                      nxh+4, nxh-4, -nxh+4, -nxh-4, 4*nxh+1, 4*nxh-1, -4*nxh+1, -4*nxh-1,
                      2*nxh+4, 2*nxh-4, -2*nxh+4, -2*nxh-4, 4*nxh+2, 4*nxh-2, -4*nxh+2, -4*nxh-2]

        # repeat indexshift number of degrees of freedom times
        shift_array = np.tile(indexshift, ndofsh)
        row_array = np.repeat(range(ndofsh), len(indexshift)*np.ones(ndofsh, dtype=int))
        column_array = row_array + shift_array
        value_array = np.tile(valloc, ndofsh)
        id_array = (column_array >= 0) & (column_array<ndofsh)

        # delete all entries corresponding to shifts smaller than 0 or bigger than ndofsh
        column_array = column_array[id_array]
        value_array = value_array[id_array]
        row_array = row_array[id_array]

        # delete artificial columns that were added in order to be able to work with "shifts"
        # note that we added 4 columns on the right hand side of the rectangle and 4 is exactly the "radius" of the
        # local stencil

        # artificial indicees of the extra columns
        range_ny = np.asarray(range(ny))
        art_indices = np.concatenate([range_ny*nxh+nx, range_ny*nxh+(nx+1), range_ny*nxh+(nx+2), range_ny*nxh+(nx+3)],
                                     axis = 0)
        id_art = np.ones(ndofsh)
        id_art[art_indices] = np.zeros(len(art_indices))
        id_art = id_art.astype(bool)

        # translation extended dofs to dofs
        trans = np.asarray(range(ndofsh))
        trans = trans[id_art]
        transinv = -1.0*np.ones(ndofsh)
        transinv[trans] = np.asarray(range(ndofs))

        # delete corresponding entries
        id_artcol = [id_art[col]==True for col in column_array]
        column_array = column_array[id_artcol]
        value_array = value_array[id_artcol]
        row_array = row_array[id_artcol]
        id_artrow = [id_art[row]==True for row in row_array]
        column_array = column_array[id_artrow]
        value_array = value_array[id_artrow]
        row_array = row_array[id_artrow]

        # transform extended dofs to dofs
        column_array = transinv[column_array]
        row_array = transinv[row_array]

        # sparse matrix
        M = sps.csr_matrix((value_array, (row_array, column_array)), shape=(ndofs, ndofs))

        # boundary corrections
        bdof = self.__boundary_classification()
        val = self.__values_boundary_correction()

        indx = []
        valx = []
        for b in bdof:
            indx += bdof[b]
            valx += (val[b]*np.ones(len(bdof[b]))).tolist()

        indx = np.asarray(indx)
        valx = np.asarray(valx)

        # sparse matrix that contains boundary correction terms
        Mmod = sps.csr_matrix((valx, (indx, indx)), shape=(ndofs, ndofs))

        M = M + Mmod

        return M

    def __inverse_of_square_root(self, M):
        """
        computes M^(-1/2); computational bottleneck of the algorithm; only needed because we work with ipopt
        which does not support to work with the "correct" inner product; see discussion in Sec.7.7;
        actually it would suffice to compute a decomposition M = L^T L and work with L instead of M^(-1/2)
        """
        return fractional_matrix_power(M.todense(), -0.5)


    def __values_boundary_correction(self):
        val={}
        for s in ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']:
            s2 = 'e1' + s
            val[s]= self.loc[s2] - self.loc['e1a']
        return val

    def __boundary_classification(self):
        """
        Here, we manually perform the boundary classification in Fig. 7.1 by assigning the degrees of freedom to
        the corresponding classes.
        """

        # degrees of freedom for different boundary parts
        # here: manually implemented for a simple 2d example
        nx = self.nx
        ny = self.ny

        bdof = {}
        bdof['b'] = np.concatenate([np.linspace(3*nx+4, 4*nx-5, nx-8), np.linspace((ny-4)*nx+4, (ny-3)*nx-5, nx-8),
                                    np.linspace(4, ny-5, ny-8)*nx +3, np.linspace(5,ny-4,ny-8)*nx-4]).astype(int).tolist()
        bdof['d'] = np.concatenate([np.linspace(2*nx+4, 3*nx-5, nx-8), np.linspace((ny-3)*nx+4, (ny-2)*nx-5, nx-8),
                                    np.linspace(4, ny-5, ny-8)*nx +2, np.linspace(5, ny-4, ny-8)*nx-3]).astype(int).tolist()
        bdof['g'] = np.concatenate([np.linspace(nx+4, 2*nx-5,nx-8), np.linspace((ny-2)*nx+4, (ny-1)*nx-5, nx-8),
                                    np.linspace(4,ny-5,ny-8)*nx+1, np.linspace(5, ny-4,ny-8)*nx-2]).astype(int).tolist()
        bdof['k'] = np.concatenate([np.linspace(4,nx-5, nx-8), np.linspace((ny-1)*nx+4, ny*nx-5, nx-8),
                                    np.linspace(4, ny-5, ny-8)*nx, np.linspace(5,ny-4,ny-8)*nx -1],axis =0).astype(int).tolist()
        bdof['c'] = [3*nx+3, 4*nx-4, (ny-4)*nx+3, (ny-3)*nx-4]
        bdof['e'] = [2*nx+3, 3*nx-4, (ny-3)*nx+3, (ny-2)*nx-4, 3*nx+2, 4*nx-3, (ny-4)*nx+2, (ny-3)*nx-3]
        bdof['f'] = [2*nx+2, 3*nx-3, (ny-3)*nx+2, (ny-2)*nx-3]
        bdof['h'] = [nx+3, 2*nx-4, (ny-2)*nx+3, (ny-1)*nx-4, 3*nx+1, 4*nx-2, (ny-4)*nx+1, (ny-3)*nx-2]
        bdof['i'] = [nx +2, 2*nx-3, (ny-2)*nx+2, (ny-1)*nx-3, 2*nx+1, 3*nx-2, (ny-3)*nx+1, (ny-2)*nx-2]
        bdof['j'] = [nx+1, 2*nx-2, (ny-2)*nx+1, (ny-1)*nx-2]
        bdof['l'] = [3, nx-4, (ny-1)*nx+3, ny*nx-4, 3*nx, 4*nx-1, (ny-4)*nx, (ny-3)*nx-1]
        bdof['m'] = [2, nx-3, (ny-1)*nx+2, ny*nx-3, 2*nx, 3*nx-1, (ny-3)*nx, (ny-2)*nx-1]
        bdof['n'] = [1, nx-2, nx, 2*nx-1, ny*nx-2, (ny-1)*nx +1, (ny-2)*nx, (ny-1)*nx-1]
        bdof['o'] = [0, nx-1, (ny-1)*nx, ny*nx-1]
        return bdof

"""
from dof_to_rho import dof_to_rho

    def test_boundary_classification(self, DG0FunctionSpace):
        DofToRho = dof_to_rho(DG0FunctionSpace)
        k = len(Function(DG0FunctionSpace).vector()[:])

        xi = np.ones(int(k / 2))
        bdof = self.__boundary_classification()

        i = 1
        for s in ['b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o']:
            i += 1
            xi[bdof[s]] = i * np.ones(len(bdof[s]))

        xif = DofToRho.eval(xi)

        file = File('./Output/xif.pvd')
        file << xif
        pass
"""