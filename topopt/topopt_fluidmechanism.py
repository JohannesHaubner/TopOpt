from dolfin import *
from dolfin_adjoint import *
import numpy as np
from scipy import io
import ufl

set_log_level(LogLevel.ERROR)

from preprocessing import Preprocessing
from ipopt_solver import IPOPTSolver, IPOPTProblem
import Hs_regularization as Hs_reg

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

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return conditional(gt(rho, 1.0),0.0, conditional(gt(rho, -1.0),
                                                     alphabar*(-1.0/16*rho**4 + 3.0/8*rho**2 -0.5*rho + 3.0/16),
                                                     -1.0*alphabar*rho))

N = 40
delta = 5.0  # The aspect ratio of the domain, 1 high and \delta wide
V = 1.0/3 * delta  # want the fluid to occupy 1/3 of the domain
mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), int(delta*N), N))

controls_file = File('../Output/final_controls_new_' + str(N) +'.pvd')

class DOI(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], (2.5)) and between(x[1], (0.475, 0.525))

class Inflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class Outflow(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], delta) and on_boundary

class Noslip(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.) or near(x[1], 1.)

doi = DOI()
inflow = Inflow()
outflow = Outflow()
noslip = Noslip()
marker = cpp.mesh.MeshFunctionSizet(mesh, 1)
marker.set_all(0)
doi.mark(marker, 1)
inflow.mark(marker, 2)
outflow.mark(marker, 3)
noslip.mark(marker, 4)

file = File("../Output/mesh.pvd")
file << marker
ds = Measure("dS", domain=mesh, subdomain_data=marker)

# test if alpha does the correct thing
#P_h = FiniteElement("CG", mesh.ufl_cell(), 1)
#P = FunctionSpace(mesh, P_h)
#c = interpolate(Expression("-4+8*x[0]", degree=1), P)
#testfile = File('./Output/c.pvd')
#v = TestFunction(P)
#vh = assemble(alpha(c)*v*dx)
#c.vector()[:] = vh[:]
#testfile << c

A = FunctionSpace(mesh, "DG", 0)        # control function space

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

        values[0] = 4.*x[1]*(1.0 - x[1])

    def value_shape(self):
        return (2,)

def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = Function(W)
    (u, p) = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    F = (alpha(rho) * inner(u, v) * dx + inner(grad(u), grad(v)) * dx +
         inner(grad(p), v) * dx  + inner(div(u), q) * dx)
    bc1 = DirichletBC(W.sub(0), InflowOutflow(degree=2), marker, 2)
    bc2 = DirichletBC(W.sub(1), Constant(0.0), marker, 3)
    bc3 = DirichletBC(W.sub(0), Constant((0., 0.)), marker, 4)
    bc = [bc1, bc2, bc3]
    solve(lhs(F) == rhs(F), w, bcs=bc)

    return w

def save_control(x0, controls_file, index=-1, J = None): #TODO
    rho = preprocessing.dof_to_control(x0)
    rho.rename("density", "density")
    print('objective function value J', J(rho))
    controls_file << rho
    if index +1:
        filename = '../Output/matlab_controls_' + str(N) + '_' + str(index +1) + '.mat'
        io.savemat(filename, mdict={'data': x0})
    pass

if __name__ == "__main__":
    x0 = (2.*V/delta -1)*np.ones(int(k/2))

    # preprocessing class which contains dof_to_control-mapping
    weighting = 0.1 # consider L2-mass-matrix + weighting * Hs-matrix
    sigma = 7./16
    preprocessing = Preprocessing(N, B)
    inner_product_matrix = Hs_reg.AssembleHs(N,delta,sigma).get_matrix(weighting)

    rho = preprocessing.dof_to_control(x0)

    # reference value
    wref = forward(Constant(1.0))
    uref, pref = wref.split(deepcopy=True)
    ref = assemble(0.5 * mu * inner(grad(uref), grad(uref)) * dx)

    # get reduced objective function: rho --> j(rho)
    set_working_tape(Tape())
    w   = forward(rho)
    (u, p) = split(w)

    controls = File("../Output/control_iterations_guess_" + str(N) +".pvd")
    allctrls = File("../Output/allcontrols_" + str(N) + ".pvd")
    rho_viz = Function(A, name="ControlVisualisation")


    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz

    # objective function
    J = assemble( inner(avg(u), Constant((1., 0)))*ds(1)) #1e2
    # J += 1e-3*assemble(0.5 * inner(alpha(rho) * u, u) * dx + 0.5 * mu * inner(grad(u), grad(u)) * dx) #1e-3 works good
    # penalty term in objective function
    J2 = assemble(ufl.Max(rho - 1.0, 0.0)**2 *dx + ufl.Max(-rho - 1.0, 0.0)**2 *dx)
    m = Control(rho)
    #
    Js = [J, J2]

    Jeval = ReducedFunctional(J, m)

    # constraints
    v = 1.0 /V * assemble((0.5 * (rho + 1)) * dx) - 1.0 # volume constraint
    s = assemble( 1.0/delta*(rho*rho - 1.0) * dx)         # spherical constraint
    g = assemble(0.5 * inner(alpha(rho) * u, u) * dx + 0.5 * mu * inner(grad(u), grad(u)) * dx) / (130 * ref) - 1.0
    constraints = [ReducedFunctional(v,m), ReducedFunctional(s,m), ReducedFunctional(g,m)]
    bounds = [[0.0, 0.0],[-1.0, 0.0],[-1e6, 0.0]] # [[lower bound vc, upper bound vc],[lower bound sc, upper bound sc]]

    # scaling
    scaling_Js = [1.0, 0.0]  # objective for optimization: scaling_Jhat[0]*Jhat[0]+scaling_Jhat[1]*Jhat[1]
    scaling_constraints = [1.0, 1.0, 1.0]  # scaling of constraints for Ipopt

    # for performance reasons we first add J and J2 and hand the sum over to the IPOPT solver
    J_ = 0
    for i in range(len(Js)):
        J_ += Js[i] * scaling_Js[i]
    Jhat = [ReducedFunctional(J_, m, eval_cb_post=eval_cb)]

    reg = 1e-6                     # regularization parameter

    # problem
    problem = IPOPTProblem(Jhat, [1.0], constraints, scaling_constraints, bounds,
                           preprocessing, inner_product_matrix, reg)
    ipopt = IPOPTSolver(problem)

    #ipopt.test_objective(len(x0))
    #ipopt.test_constraints(len(x0), 1, option=1)

    x0 = ipopt.solve(x0)

    save_control(x0, controls_file, 0, J = Jhat[0])

    # different weights for H_sigma matrix
    weight = [0.01, 0.01, 0.001]
    # different penalization parameters
    eta = [0.4, 2, 10]

    for j in range(len(eta)):
        # bounds for the constraints
        bounds = [[-1e6, 0.0], [0.0, 0.0], [-1e6, 0.0]]

        reg = 1e-6

        # update inner product
        weighting = weight[j]  # consider L2-mass-matrix + weighting * Hs-matrix
        inner_product_matrix = Hs_reg.AssembleHs(N, delta, sigma).get_matrix(weighting)

        scaling_Js = [1.0, eta[j]]

        # for performance reasons we first add J and J2 and hand the sum over to the IPOPT solver
        J_ = 0
        for i in range(len(Js)):
            J_ += Js[i] * scaling_Js[i]
        Jhat = [ReducedFunctional(J_, m, eval_cb_post=eval_cb)]

        # move x0 onto sphere
        x0 = preprocessing.move_onto_sphere(x0, V, delta)

        # solve optimization problem
        problem = IPOPTProblem(Jhat, [1.0], constraints, scaling_constraints, bounds, preprocessing,
                               inner_product_matrix, reg)
        ipopt = IPOPTSolver(problem)

        x0 = ipopt.solve(x0)
        save_control(x0, controls_file, j+1, J = Jeval)
