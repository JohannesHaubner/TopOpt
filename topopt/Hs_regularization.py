from dolfin import *
from dolfin_adjoint import *
import numpy as np
from quadrature.quadrature import get_weights
import scipy.sparse as sps
import scipy.sparse as sparse
import pickle

class AssembleHs:
    def __init__(self, N, delta, sigma):
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

        """

        # entries of local stencil
        self.sigma = sigma
        file = get_weights(sigma)
        self.ints = pickle.load(file)

        self.h = 1.0/N
        self.nx = int(delta*N)
        self.ny = int(N)
        self.loc = self.__get_entries_of_local_stencils()
        self.Hs_glob_matrix = self.__get_global_matrix()
        self.L2_glob_matrix = self.__get_L2_matrix()


    def get_matrix(self, weighting):
        """
        returns L^2-matrix + weighting*H^s-matrix
        """
        return self.with_coo(self.L2_glob_matrix, weighting*self.Hs_glob_matrix)

    def __get_entries_of_local_stencils(self):
        """
        The values were computed using quadrature rules based on the integrals presented in Sec.7 of the manuscript
        """
        h = self.h
        sigma = self.sigma
        prefac = -2*h**(2.-2*sigma)

        loc = {}

        loc['e2'] = prefac*(2./(1. - 2.*sigma) *self.ints['int2_1'][0]
                            + 2.0/(2 - 2*sigma)**self.ints['int2_2'][0])
        loc['e3'] = prefac*(4./(2. - 2.*sigma) *self.ints['int3'][0])
        loc['e4'] = prefac*self.ints['int4'][0]
        loc['e5'] = prefac*self.ints['int5'][0]
        loc['e6'] = prefac*self.ints['int6'][0]
        loc['e7'] = prefac*self.ints['int7'][0]
        loc['e8'] = prefac*self.ints['int8'][0]
        loc['e9'] = prefac*self.ints['int9'][0]
        loc['e10'] = prefac*self.ints['int10'][0]
        loc['e11'] = prefac*self.ints['int11'][0]
        loc['e12'] = prefac*self.ints['int12'][0]
        loc['e13'] = prefac*self.ints['int13'][0]

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

    @staticmethod
    def with_coo(x, y):
        # add two csc matrices
        # https://stackoverflow.com/questions/37231163/adding-two-csc-sparse-matrices-of-different-shapes-in-python
        x = x.tocoo()
        y = y.tocoo()
        d = np.concatenate((x.data, y.data))
        r = np.concatenate((x.row, y.row))
        c = np.concatenate((x.col, y.col))
        C = sparse.coo_matrix((d, (r, c)))
        return C.tocsc()

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
                                    np.linspace(4, ny-5, ny-8)*nx +3, np.linspace(5,ny-4,ny-8)*nx-4])\
            .astype(int).tolist()
        bdof['d'] = np.concatenate([np.linspace(2*nx+4, 3*nx-5, nx-8), np.linspace((ny-3)*nx+4, (ny-2)*nx-5, nx-8),
                                    np.linspace(4, ny-5, ny-8)*nx +2, np.linspace(5, ny-4, ny-8)*nx-3])\
            .astype(int).tolist()
        bdof['g'] = np.concatenate([np.linspace(nx+4, 2*nx-5,nx-8), np.linspace((ny-2)*nx+4, (ny-1)*nx-5, nx-8),
                                    np.linspace(4,ny-5,ny-8)*nx+1, np.linspace(5, ny-4,ny-8)*nx-2])\
            .astype(int).tolist()
        bdof['k'] = np.concatenate([np.linspace(4,nx-5, nx-8), np.linspace((ny-1)*nx+4, ny*nx-5, nx-8),
                                    np.linspace(4, ny-5, ny-8)*nx, np.linspace(5,ny-4,ny-8)*nx -1],axis =0)\
            .astype(int).tolist()
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
