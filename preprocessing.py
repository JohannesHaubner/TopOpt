from dolfin import *
from dolfin_adjoint import *
import Hs_regularization as reg
import numpy as np
from scipy.sparse.linalg import spsolve

class Preprocessing:
    def __init__(self, N, delta, FunctionSpaceDG0, weighting, sigma):
        """
        we assume the mesh size to be uniform in x and y direction and
        take advantage of the fact that the numbering of the triangular cells of the rectangular domain is:

        |  /                |  /                |  /
        ____________________________________________
        | 2*int(delta*N) /  | 2*int(delta*N) /  |

        |    +1      /      |    +3      /      |

        |       / 0 +       |       / 2 +       |

        |  / 2*int(delta*N) |  / 2*int(delta*N) |  /
        ____________________________________________
        |                /  |                /  |

        |     1      /      |     3      /      |

        |       / 0         |       / 2         |

        |  /                |  /                |  /
        ____________________________________________

        whereas the degrees of freedom correspond to a DG0 discretization on a uniform quadrilateral mesh with the
        numbering:

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
        self.DG0 = FunctionSpaceDG0
        self.k = len(Function(self.DG0).vector()[:])
        self.h = 1./N
        regularization = reg.AssembleHs(N, delta, 0.5*sigma)
        self.matrix = 1./self.h*regularization.get_matrix(weighting)

    def transformation(self, x):
        """
        x --> (1./h * H^(sigma/2)_matrix)^(-1)*x
        """
        return spsolve(self.matrix, x)

    def transformation_chainrule(self, djy):
        """
        djy --> (1./h * H^(sigma/2)_matrix)^(-T)*djy
        """
        return spsolve(self.matrix.transpose(), djy)

    def dof_to_rho(self, x):
        """
        map vector of degrees of freedom to function on triangular mesh
        degrees of freedom live on quadrilateral 2d mesh, whereas rho lives on uniform triangular 2d mesh
        """
        array = np.repeat(np.array(x), np.array(2*np.ones(len(x),dtype=int)))
        func = Function(self.DG0)
        func.vector()[:] = array
        return func

    def dof_to_rho_chainrule(self, djy, option=1):
        """
        chainrule of dof_to_rho
        """
        if option ==2:
            return djy[::2]+djy[1::2]
        else:
            return djy.vector()[::2]+djy.vector()[1::2]

    def initial_point_trafo(self, x0):
        """
        needs to be done since we apply the affine linear transformation afterwards
        """
        return self.matrix.dot(x0)

    def move_onto_sphere(self, y0, V, delta):
        """
        y0 describes the density function defined on the uniform quadrilateral mesh,
        described in Sec. 7.6
        """
        y00 = (2.*V/delta-1.)*np.ones(len(y0))
        deltay = y0-y00
        ub = np.ones(len(y0))
        int1 = np.dot(ub,ub)
        int2 = np.dot(y00,y00)
        int3 = np.dot(deltay, deltay)
        t = np.sqrt((int1-int2)/int3)

        return y00 +t*deltay

    def move_control_onto_sphere(self, x0, V, delta):
        """
        move control x0 (ndarray of dofs on quadrilateral mesh) onto sphere that is defined via V and delta
        """
        y0 = self.transformation(x0)
        y0 = self.move_onto_sphere(y0, V, delta)
        return self.initial_point_trafo(y0)

