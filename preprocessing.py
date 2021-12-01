from dolfin import *
from dolfin_adjoint import *
import Hs_regularization as reg
import numpy as np

class Preprocessing:
    def __init__(self, N, delta, FunctionSpaceDG0):
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
        self.regularization = reg.Regularization(N, delta)

    def transformation(self, x):
        return self.regularization.transform(x)

    def transformation_chainrule(self, djy):
        return self.regularization.transform_chainrule(djy)

    def dof_to_rho(self, x):
        array = np.repeat(np.array(x), np.array(2*np.ones(len(x),dtype=int)))
        func = Function(self.DG0)
        func.vector()[:] = array
        return func

    def dof_to_rho_chainrule(self, djy, option=1):
        if option ==2:
            return djy[::2]+djy[1::2]
        else:
            return djy.vector()[::2]+djy.vector()[1::2]

    def initial_point_trafo(self, x0):
        """
        needs to be done since we apply the affine linear transformation afterwards
        """
        return self.regularization.initial_point_trafo(x0)
