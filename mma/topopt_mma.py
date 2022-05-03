from dolfin import *
from dolfin_adjoint import *
from pyadjoint.enlisting import Enlist
import numpy as np
from scipy import io

import ufl

set_log_level(LogLevel.ERROR)

from pathlib import Path
here = Path(__file__).parent
import sys
sys.path.insert(0, str(here.parent) + "/topopt")

from mma_solver import MMASolver, MMAProblem

try:
    from pyadjoint import ipopt  # noqa: F401
except ImportError:
    print("""This example depends on IPOPT and Python ipopt bindings. \
  When compiling IPOPT, make sure to link against HSL, as it \
  is a necessity for practical problems.""")
    raise


# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

mu = Constant(1.0)                   # viscosity
alphaunderbar = 0                    # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho, q):
    """Inverse permeability as a function of rho, equation (40)"""
    return (alphaunderbar-alphabar)*rho*(1+q)/(rho+q) + alphabar

N = 40
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
V = 1.0/3 * delta  # want the fluid to occupy 1/3 of the domain
mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), int(delta*N), N))

controls_file = File('../Output/final_controls_' + str(N) +'.pvd')

# test if alpha does the correct thing
#P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
#P = FunctionSpace(mesh, P_h)
#c = interpolate(Expression("-4+8*x[0]", degree=1), P)
#testfile = File('./Output/c.pvd')
#v = TestFunction(P)
#vh = assemble(alpha(c)*v*dx)
#c.vector()[:] = vh[:]
#testfile << c

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

B = FunctionSpace(mesh, "DG", 0)          # control function space
b = Function(B)
k = len(b.vector()[:])
b.vector()[:] = range(k)

#file = File("./Output/b_ved.pvd")
#file << b


# Define the boundary condition on velocity

class InflowOutflow(UserExpression):
    def eval(self, values, x):
        values[1] = 0.0
        values[0] = 0.0
        l = 1.0/6.0
        gbar = 1.0

        if x[0] == 0.0 or x[0] == delta:
            if (1.0/4 - l/2) < x[1] < (1.0/4 + l/2):
                t = x[1] - 1.0/4
                values[0] = gbar*(1 - (2*t/l)**2)
            if (3.0/4 - l/2) < x[1] < (3.0/4 + l/2):
                t = x[1] - 3.0/4
                values[0] = gbar*(1 - (2*t/l)**2)

    def value_shape(self):
        return (2,)

# pressure BC
class PressureB(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], (0.0)) and near(x[1], (0.0))
pressureB = PressureB()


def forward(rho, q_):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    #w = Function(W)
    w = interpolate(Constant((0., 0., 0.)), W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (alpha(rho, q_) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bc = [DirichletBC(W.sub(0), InflowOutflow(degree=2), "on_boundary"),
          DirichletBC(W.sub(1), Constant(0.0), pressureB, method='pointwise')]
    solve(lhs(F) == rhs(F), w, bcs=bc)

    return w

def save_control(x0, controls_file, index=-1, J = None):
    rho = Function(B)
    rho.vector()[:] = x0
    rho.rename("density", "density")
    print('objective function value J', J(rho))
    controls_file << rho
    if index + 1:
        filename = '../Output/matlab_controls_' + str(N) + '_' + str(index + 1) + '.mat'
        io.savemat(filename, mdict={'data': x0})
    pass

if __name__ == "__main__":
    x0 = (2.*V/delta -1)*np.ones(int(k))

    # preprocessing class which contains dof_to_control-mapping
    weighting = 1.  # consider L2-mass-matrix + weighting * Hs-matrix
    sigma = 7./16

    controls = File("../Output/control_iterations_guess" + str(N) + ".pvd")
    allctrls = File("../Output/allcontrols_" + str(N) + ".pvd")
    rho_viz = Function(B, name="ControlVisualisation")


    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz

    for q in [0.01, 0.1]:
        set_working_tape(Tape())

        rho = Function(B)
        rho.vector()[:] = x0
        # get reduced objective function: rho --> j(rho)
        w = forward(rho, q)
        (u, p) = split(w)

        # objective function
        J = assemble(0.5 * inner(alpha(rho, q) * u, u) * dx + 0.5 * mu * inner(grad(u), grad(u)) * dx)

        m = Control(rho)
        Jhat = [ReducedFunctional(J, m, eval_cb_post=eval_cb)]

        # constraints
        v = 1.0 /V * assemble(rho * dx) - 1.0  # volume constraint

        constraints = [ReducedFunctional(v,m)]

        # scaling
        scaling_Jhat = [1.0, 0.0]          # objective for optimization: scaling_Jhat[0]*Jhat[0]+scaling_Jhat[1]*Jhat[1]
        scaling_constraints = [1.0]        # scaling of constraints

        # simple bound constraints
        low_bound = np.zeros(len(rho.vector()[:]))
        up_bound = np.ones(len(rho.vector()[:]))

        # problem
        problem = MMAProblem(Jhat, scaling_Jhat, constraints, scaling_constraints, [low_bound, up_bound])
        parameters = {}
        if q == 0.01:
            parameters["maxit"] = 30
        elif q == 0.1:
            parameters["maxit"] = 300
        mma = MMASolver(problem, parameters)

        #ipopt.test_objective(len(x0))
        #ipopt.test_constraints(len(x0), 1, option=1)

        x0 = mma.solve(rho.vector()[:]).T[0]

        save_control(x0, controls_file, 0, J = Jhat[0])

