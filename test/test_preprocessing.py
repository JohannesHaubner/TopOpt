import pytest
import preprocessing
import ipopt_solver
import numpy as np

from dolfin import *
from dolfin_adjoint import *

def setting(N):
    # simplified setting
    mesh = Mesh(RectangleMesh(MPI.comm_world, Point(0.0, 0.0), Point(1.0, 1.0), N, N))

    # function space
    B = FunctionSpace(mesh, "DG", 0)
    b = Function(B)

    # degrees of freedom
    k = len(b.vector()[:])

    return mesh, B, k


def test_chainrule():
    # simplified setting
    N = 10
    mesh, B, k = setting(N)
    preproc = preprocessing.Preprocessing(N, B)

    # degrees of freedom
    b = Function(B)
    k = len(b.vector()[:])

    # point
    x0 = np.ones(int(k/2))

    # perturbation direction
    ds = np.ones(int(k/2))

    # objective and derivative
    b = preproc.dof_to_control(x0)
    J = assemble(inner(b, b) * dx)
    Jhat = ReducedFunctional(J, Control(b))
    j0 = Jhat(b)
    dJ = Jhat.derivative()
    dj = preproc.dof_to_control_chainrule(dJ)

    # perturbed objective
    epslist = [1.0, 0.1, 0.01, 0.001, 0.0001]
    xi = [x0 + epslist[i]*ds for i in range(len(epslist))]
    bi = [preproc.dof_to_control(x) for x in xi]
    jlist = [Jhat(b) for b in bi]

    # first order check
    print('First order check for preprocessing.dof_to_control................................')
    order1, diff1 = ipopt_solver.IPOPTSolver.perform_first_order_check(jlist, j0, dj, ds, epslist)

    assert order1[-1] > 1.8

def test_move_onto_sphere_valueerror():
    # setting
    N = 10
    mesh, B, k = setting(N)
    delta = 1.0
    V = 0.33

    y00 = (2. * V / delta - 1.) * np.ones(int(k/2))

    with pytest.raises(ValueError):
        preprocessing.Preprocessing(N,B).move_onto_sphere(y00, V, delta)

def test_move_onto_sphere():
    # setting
    N = 10
    mesh, B, k = setting(N)
    delta = 1.0
    V = 0.33
    preproc = preprocessing.Preprocessing(N,B)

    # projection of 0
    y00 = (2. * V / delta - 1.) * np.ones(int(k/2))

    # perturbation
    eps = 0.02
    ds = 0.02*np.ones(len(y00))
    ds[range(int(k/4))] = -1.0*ds[range(int(k/4))]
    y0 = y00 + ds

    y = preproc.move_onto_sphere(y0, V, delta)

    # check spherical constraint
    rho0 = preproc.dof_to_control(y)
    assert assemble((rho0*rho0 - 1.0)*dx) < 1e-14
