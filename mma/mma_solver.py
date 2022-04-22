from dolfin import *
from dolfin_adjoint import *
from copy import deepcopy
import numpy as np
import backend

from pyadjoint.optimization.optimization_solver import OptimizationSolver
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

import logging
import sys
import os

try:
    from MMA_Python.Code.MMA import gcmmasub, subsolv, kktcheck, mmasub
except ImportError:
    print("MMA solver not found. Requires https://github.com/arjendeetman/GCMMA-MMA-Python ")
    raise


import matplotlib.pyplot as plt

class MMAProblem:
    def __init__(self, Jhat, scaling_Jhat, constraints, scaling_constraints, simple_bound_const):
        """
        Jhat:                   list of different objective function summands
        scaling_Jhat:           list of floats with same length as Jhat that specifies how the sum of the different
                                objective function summands is taken
        constraints:            list of inequality constraints (constraints <= 0)
        scaling_constraints:    list with same length as constraints that specifies scaling of constraints
        simple_bound_const:     two arrays with length Control giving upper and lower bound constraints on control
        """
        self.Jhat = [ReducedFunctionalNumPy(Jhat[i]) for i in range(len(Jhat)) ]
        self.scaling_Jhat = scaling_Jhat
        self.constraints = [ReducedFunctionalNumPy(constraints[i]) for i in range(len(constraints))]
        self.scaling_constraints = scaling_constraints
        self.simple_bound_const = simple_bound_const

    def __call__(self, x0):
        if len(x0) != len(self.Jhat[0].get_controls()):
            raise ValueError('length of control not equal to length of argument')
        # evaluation of objective
        f0val = 0
        for i in range(len(self.Jhat)):
            f0val += self.Jhat[i](x0)*self.scaling_Jhat[i]
        # evaluation of gradient of objective
        df0dx = np.zeros(len(x0))
        for i in range(len(self.Jhat)):
            df0dx += self.scaling_Jhat[i] * self.Jhat[i].derivative(forget=False, project=False)
        df0dx = np.asarray([df0dx]).T
        # evaluation of inequality constraints
        fval = []
        for i in range(len(self.constraints)):
            fival = self.scaling_constraints[i]*self.constraints[i](x0)
            if not isinstance(fival, float):
                raise NotImplementedError('Function value of constraint needs to be float. '
                                          'Vector valued constraints are not implemented yet.')
            fval.append([self.scaling_constraints[i]*self.constraints[i](x0)])
        fval = np.asarray(fval)
        # evaluation of inequality jacobian
        dfdx = np.asarray([])
        for i in range(len(self.constraints)):
            if i == 0:
                dfdx = self.constraints[i].derivative().reshape((len(x0), 1)).T
            else:
                dfidx = self.constraints[i].derivative().reshape((len(x0), 1)).T
                dfdx = np.concatenate((dfdx, dfidx))
        return f0val, df0dx, fval, dfdx


class MMASolver(OptimizationSolver):
    def __init__(self, problem, parameters=None):
        self.problem = problem
        self.maxoutit = 100
        #self.lin_const = False
        self.globalize = False

    def solve(self, rho0):
        """
        code snippet taken from
        https://github.com/arjendeetman/GCMMA-MMA-Python/blob/master/Code/MMA_TOY2.py
        """
        # Logger
        path = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(path, "MMA.log")
        logger = self.setup_logger(file)
        logger.info("Started\n")
        # Set numpy print options
        np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
        # Initial settings
        m = len(self.problem.constraints)
        n = len(rho0)
        epsimin = 0.0000001
        eeen = np.ones((n, 1))
        eeem = np.ones((m, 1))
        zeron = np.zeros((n, 1))
        zerom = np.zeros((m, 1))
        xval = rho0.reshape((n,1))
        xold1 = xval.copy()
        xold2 = xval.copy()
        xmin = self.problem.simple_bound_const[0].reshape((n, 1))
        xmax = self.problem.simple_bound_const[1].reshape((n, 1))
        low = xmin.copy()
        upp = xmax.copy()
        move = 1.0
        c = 1000 * eeem
        d = eeem.copy()
        a0 = 1
        a = zerom.copy()
        outeriter = 0
        maxoutit = self.maxoutit
        kkttol = 0
        # Calculate function values and gradients of the objective and constraints functions
        if outeriter == 0:
            f0val, df0dx, fval, dfdx = self.problem(xval.T[0])
            innerit = 0
            outvector1 = np.concatenate((np.array([outeriter]), xval.flatten()))
            outvector2 = np.concatenate((np.array([f0val]), fval.flatten()))
            # Log
            logger.info("outvector1 = {}".format(outvector1))
            logger.info("outvector2 = {}\n".format(outvector2))
        # The iterations starts
        kktnorm = kkttol + 10
        outit = 0
        while (kktnorm > kkttol) and (outit < maxoutit):
            outit += 1
            outeriter += 1
            # The MMA subproblem is solved at the point xval:
            globalize = self.globalize
            if not globalize:
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = \
                mmasub(m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d,
                       move)
            else:
                raa0 = 0.01
                raa = 0.01 * eeem
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = \
                    gcmmasub(m, n, outeriter, epsimin, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx,
                        a0, a, c, d)
            # Some vectors are updated:
            xold2 = xold1.copy()
            xold1 = xval.copy()
            xval = xmma.copy()
            # Re-calculate function values and gradients of the objective and constraints functions
            f0val, df0dx, fval, dfdx = self.problem(xval.T[0])
            # The residual vector of the KKT conditions is calculated
            residu, kktnorm, residumax = \
                kktcheck(m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
            outvector1 = np.concatenate((np.array([outeriter]), xval.flatten()))
            outvector2 = np.concatenate((np.array([f0val]), fval.flatten()))
            # Log
            logger.info("outvector1 = {}".format(outvector1))
            logger.info("outvector2 = {}".format(outvector2))
            logger.info("kktnorm    = {}\n".format(kktnorm))
        # Final log
        logger.info("Finished")

        return xval

    # Setup logger
    @staticmethod
    def setup_logger(logfile):
        # Create logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        # Create console handler and set level to debug
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create file handler and set level to debug
        fh = logging.FileHandler(logfile)
        fh.setLevel(logging.DEBUG)
        # Add formatter to ch and fh
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # Add ch and fh to logger
        logger.addHandler(ch)
        logger.addHandler(fh)
        # Open logfile and reset
        with open(logfile, 'w'): pass
        # Return logger
        return logger