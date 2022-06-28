from dolfin import *
from dolfin_adjoint import *
import numpy as np
from pyadjoint.enlisting import Enlist


class Preprocessing:
    def __init__(self, N, FunctionSpaceDG0, parameters=None):
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
        self.parameters = False
        if parameters != None:
            self.parameters = True
            self.range_max = parameters["design_len"]


    def dof_to_control(self, x):
        """
        map vector of degrees of freedom to function on triangular mesh
        degrees of freedom live on quadrilateral 2d mesh, whereas rho lives on uniform triangular 2d mesh
        """
        x0 = x
        if self.parameters:
            x = x[:self.range_max]
        array = np.repeat(np.array(x), np.array(2*np.ones(len(x),dtype=int)))
        func = Function(self.DG0)
        func.vector()[:] = array
        array = func.vector()[:]
        array = np.append(array, x0[self.range_max:])
        return array

    def dof_to_control_chainrule(self, djy, option=1):
        """
        chainrule of dof_to_control
        """
        djy0 = djy
        if self.parameters:
            djy = djy[:2*self.range_max]
        if option ==2:
            djy_ = djy[::2]+djy[1::2]
            if self.parameters:
                djy_ = np.append(djy_, djy0[2*self.range_max:])
            return djy_
        else:
            djy_ = djy.vector()[::2]+djy.vector()[1::2]
            if self.parameters:
                djy_ = np.append(djy_, djy0[2*self.range_max:])
            return djy_

    def move_onto_sphere(self, y0, V, delta):
        """
        y0 describes the density function defined on the uniform quadrilateral mesh,
        described in Sec. 7.6
        """
        if self.parameters:
            y0 = y0[:self.range_max]
        y00 = (2.*V/delta-1.)*np.ones(len(y0))
        if (y00 - y0).all() < 1e-14:
            raise ValueError("Only works for non-Constant y0")
        deltay = y0-y00
        ub = np.ones(len(y0))
        int1 = np.dot(ub,ub)
        int2 = np.dot(y00,y00)
        int3 = np.dot(deltay, deltay)
        t = np.sqrt((int1-int2)/int3)

        y_ = y00 +t*deltay

        if self.paramters:
            y_ = np.append(y_, y0[self.range_max:])

        return y_

