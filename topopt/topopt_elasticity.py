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

mu = Constant(0.38)
lam = Constant(0.58)
alphaunderbar = 1e-3                 # parameter for \alpha
alphabar = 1.0 - alphaunderbar       # parameter for \alpha
q = Constant(0.01) # q value that controls difficulty/discrete-valuedness of solution

def alpha(rho):
    """Inverse permeability as a function of rho, equation (40)"""
    return conditional(gt(rho, 1.0),0.0, conditional(gt(rho, -1.0),
                                                     alphabar*(-1.0/16*rho**4 + 3.0/8*rho**2 -0.5*rho + 3.0/16),
                                                     -1.0*alphabar*rho))+alphaunderbar

N = 40
delta = 2.0  # The aspect ratio of the domain, 1 high and \delta wide
V = 0.5 * delta  # want the fluid to occupy 1/2 of the domain

mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(delta, 1.0), int(delta*N), N))

controls_file = File('../Output/final_controls_new_' + str(N) +'.pvd')

class Boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class FixedBC(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary

class ForceBC(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[1], (0.45, 0.55)) and near(x[0], delta) and on_boundary

boundary = Boundary()
fixed = FixedBC()
force = ForceBC()

marker = cpp.mesh.MeshFunctionSizet(mesh, 1)
marker.set_all(0)
boundary.mark(marker, 1)
fixed.mark(marker, 2)
force.mark(marker, 3)

file = File("mesh_elasticity.pvd")
file << marker
ds = Measure("ds", domain=mesh, subdomain_data=marker)

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
W = FunctionSpace(mesh, U_h)          

B = FunctionSpace(mesh, "DG", 0)
b = Function(B)
k = len(b.vector()[:])
b.vector()[:] = range(k)

#file = File("./Output/b_ved.pvd")
#file << b


# Define the boundary condition on velocity

def forward(rho):
    """Solve the forward problem for a given fluid distribution rho(x)."""
    w = Function(W)
    u = TrialFunction(W)
    v = TestFunction(W)

    I = Identity(w.geometric_dimension())
    
    epsu = Constant(0.5)*(grad(u) + grad(u).T)
    epsv = Constant(0.5)*(grad(v) + grad(v).T)
    C = alpha(rho)*(2*mu*epsu + lam * tr(epsu) * I)

    F = inner(C, epsv)*dx - inner(Constant((0,-1.)), v)*ds(3)
    bc1 = DirichletBC(W, Constant((0.,0.)), marker, 2)
    bc = bc1
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
    weighting = 0.01 # consider L2-mass-matrix + weighting * Hs-matrix
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
    u  = forward(rho)

    controls = File("./Output/elasticity_control_iterations_guess_" + str(N) +".pvd")
    allctrls = File("./Output/elasticity_allcontrols_" + str(N) + ".pvd")
    rho_viz = Function(A, name="ControlVisualisation")


    def eval_cb(j, rho):
        rho_viz.assign(rho)
        controls << rho_viz
        allctrls << rho_viz

    # objective function
    I = Identity(u.geometric_dimension())
    epsu = Constant(0.5)*(grad(u) + grad(u).T)
    C = alpha(rho)*(2*mu*epsu + lam * tr(epsu) * I)
    J = assemble( inner(C, epsu)*dx) #1e2
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
    constraints = [ReducedFunctional(v,m), ReducedFunctional(s,m)]
    bounds = [[0.0, 0.0],[-1.0, 0.0]] # [[lower bound vc, upper bound vc],[lower bound sc, upper bound sc]]

    # scaling
    scaling_Js = [1.0, 0.0]  # objective for optimization: scaling_Jhat[0]*Jhat[0]+scaling_Jhat[1]*Jhat[1]
    scaling_constraints = [1.0, 1.0, 1.0]  # scaling of constraints for Ipopt

    # for performance reasons we first add J and J2 and hand the sum over to the IPOPT solver
    J_ = 0
    for i in range(len(Js)):
        J_ += Js[i] * scaling_Js[i]
    Jhat = [ReducedFunctional(J_, m, eval_cb_post=eval_cb)]

    reg = 1e-9                     # regularization parameter

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
        bounds = [[0.0, 0.0], [0.0, 0.0]]

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