from dolfin import *
from dolfin_adjoint import *
import numpy as np

from preprocessing import Preprocessing
from ipopt_solver import IPOPTSolver

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
alphaunderbar = 2.5 * mu / (100**2)  # parameter for \alpha
alphabar = 2.5 * mu / (0.01**2)      # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

controls_file = File('./Output/final_controls.pvd')

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return conditional(gt(rho, 1.0),0.0, conditional(gt(rho, -1.0),
                                                     alphabar*(-1.0/16*rho**4 + 3.0/8*rho**2 -0.5*rho + 3.0/16),
                                                     -1.0*alphabar*rho))

N = 40
delta = 1.5  # The aspect ratio of the domain, 1 high and \delta wide
V = 1.0/3 * delta  # want the fluid to occupy 1/3 of the domain
mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), int(delta*N), N))

# test if alpha does the correct thing
#P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
#P = FunctionSpace(mesh, P_h)
#c = interpolate(Expression("-4+8*x[0]", degree=1), P)
#testfile = File('./Output/c.pvd')
#v = TestFunction(P)
#vh = assemble(alpha(c)*v*dx)
#c.vector()[:] = vh[:]
#testfile << c

A = FunctionSpace(mesh, "CG", 1)        # control function space

U_h = VectorElement("CG", mesh.ufl_cell(), 2)
P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, U_h*P_h)          # mixed Taylor-Hood function space

B = FunctionSpace(mesh, "DG", 0)
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

def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bc = DirichletBC(W.sub(0), InflowOutflow(degree=2), "on_boundary")
    solve(lhs(F) == rhs(F), w, bcs=bc)

    return w

def save_control(x0):
    z0 = preprocessing.transformation(x0)
    rho = preprocessing.dof_to_rho(z0)
    rho.rename("density", "density")
    controls_file << rho
    pass

if __name__ == "__main__":
    x0 = (2.*V/delta -1)*np.ones(int(k/2))

    # preprocessing class which contains transformation and dof_to_rho-mapping
    weighting = 0.1  # consider L2-mass-matrix + weighting * Hs-matrix
    preprocessing = Preprocessing(N,delta,B, weighting)

    x0 = preprocessing.initial_point_trafo(x0)
    y0 = preprocessing.transformation(x0)
    rho = preprocessing.dof_to_rho(y0)

    # get reduced objective function: rho --> j(rho)
    set_working_tape(Tape())
    w   = forward(rho)
    (u, p) = split(w)

    controls = File("./Output/control_iterations_guess.pvd")
    allctrls = File("./Output/allcontrols.pvd")
    rho_viz = Function(A, name="ControlVisualisation")


    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz


    J = assemble(0.5 * inner(alpha(rho) * u, u) * dx + 0.5 * mu * inner(grad(u), grad(u)) * dx)
    m = Control(rho)
    Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)

    # initial optimization problem on "ball"

    param = {}
    param["reg"] = 10.   # parameter for regularization of rho in norm equivalent to H^s-norm
    param["penal"] = 0.  # penalization of L2-violation of (-1,+1)-box-constraint
    param["vol"] = 1.    # scaling of volume constraint 1/|Omega|*assemble((0.5*(rho+1))*dx)-self.V)
    param["sphere"] = 1. # scaling of spherical constraint 1/|Omega|*h^2*(np.dot(np.ones(len(x)), np.ones(len(x)))-np.dot(np.ones(len(x)),np.ones(len(x))))
    param["relax_vol"] = [0., 0.]
    param["relax_sphere"] = [-1., 0.] #[-1,0]: ||rho||² <= 1, [0,0]: ||rho||² = 1
    param["obj"] = 1.0

    param["maxiter_IPOPT"] = 100

    ipopt = IPOPTSolver(preprocessing, k, Jhat, param, 1. / N, V)
    #ipopt.test_objective()
    #exit(0)
    #ipopt.test_constraints(option=1)
    x0 = ipopt.solve(x0)

    save_control(x0)

    # move x0 onto sphere
    x0 = preprocessing.move_control_onto_sphere(x0, V, delta)

    # preprocessing class which contains transformation and dof_to_rho-mapping
    print("reinitialize preprocessing...............................")
    weighting = 1e-2  # consider L2-mass-matrix + weighting * Hs-matrix
    preprocessing = Preprocessing(N, delta, B, weighting)

    # check if it worked:
    #y0 = preprocessing.transformation(x0)
    #print(1./N * 1./N * (np.dot(np.asarray(y0), np.asarray(y0)) - np.dot(np.ones(len(y0)), np.ones(len(y0)))))

    # adapt parameters
    param["reg"] = 10.
    param["relax_vol"] = [0., 0.]
    param["relax_sphere"] = [0., 0.]

    for eta in [1000.*3/16, 5000.*3/16, 25000.*3/16]:
        param["penal"] = eta
        ipopt = IPOPTSolver(preprocessing, k, Jhat, param, 1./N, V)
        x0 = ipopt.solve(x0)
        save_control(x0)