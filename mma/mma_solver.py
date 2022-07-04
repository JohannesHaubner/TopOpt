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
    from MMA_Python.Code.MMA import gcmmasub, subsolv, kktcheck, mmasub, asymp, concheck, raaupdate
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

    def __call__(self, x0, no_grad=False):
        if len(x0) != len(self.Jhat[0].get_controls()):
            raise ValueError('length of control not equal to length of argument')
        # evaluation of objective
        f0val = 0
        for i in range(len(self.Jhat)):
            f0val += self.Jhat[i](x0)*self.scaling_Jhat[i]
        if not no_grad:
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
        if not no_grad:
            # evaluation of inequality jacobian
            dfdx = np.asarray([])
            for i in range(len(self.constraints)):
                if i == 0:
                    dfdx = self.constraints[i].derivative().reshape((len(x0), 1)).T
                else:
                    dfidx = self.constraints[i].derivative().reshape((len(x0), 1)).T
                    dfdx = np.concatenate((dfdx, dfidx))
        if not no_grad:
            return f0val, df0dx, np.asarray(fval), dfdx
        else:
            return np.asarray([[f0val]]), np.asarray(fval)


class MMASolver(OptimizationSolver):
    def __init__(self, problem, parameters=None):
        self.problem = problem
        self.maxoutit = 100
        if parameters != None:
            self.maxoutit = parameters["maxit"]
        self.kkttol = 1e-3
        # Logger
        path = os.path.dirname(os.path.realpath(__file__))
        file = os.path.join(path, "MMA.log")
        self.logger = self.setup_logger(file)
        self.lin_const = False
        self.globalize = False

    def solve(self, rho0):
        """
        code snippet based on and taken from
        https://github.com/arjendeetman/GCMMA-MMA-Python/blob/master/Code/MMA_TOY2.py and /GCMMA_BEAM.py
        """
        logger = self.logger    # changed from setup_logger(file) that is defined in __init__
        logger.info("Started\n")
        # Set numpy print options
        np.set_printoptions(precision=4, formatter={'float': '{: 0.4f}'.format})
        # Initial settings
        m = len(self.problem.constraints) # changed from m=2 to our problem setting
        n = len(rho0)                     # changed from n= 3 to our problem setting
        epsimin = 0.0000001
        eeen = np.ones((n, 1))
        eeem = np.ones((m, 1))
        zeron = np.zeros((n, 1))
        zerom = np.zeros((m, 1))
        xval = rho0.reshape((n,1))       # changed and adapted to our problem setting
        xold1 = xval.copy()
        xold2 = xval.copy()
        xmin = self.problem.simple_bound_const[0].reshape((n, 1)) # changed and adapted to our problem setting
        xmax = self.problem.simple_bound_const[1].reshape((n, 1)) # changed and adapted to our problem setting
        low = xmin.copy()
        upp = xmax.copy()
        move = 1.0
        c = 1000 * eeem
        d = eeem.copy()
        a0 = 1
        a = zerom.copy()
        outeriter = 0
        maxoutit = self.maxoutit # changed and adapted to our problem setting
        kkttol = self.kkttol     # changed and adapted to our problem setting
        # Calculate function values and gradients of the objective and constraints functions
        if outeriter == 0:
            f0val, df0dx, fval, dfdx = self.problem(xval.T[0]) # changed and adapted to our problem setting
            if self.globalize:
                innerit = 0
                outvector1 = np.concatenate((np.array([outeriter, innerit, f0val]), fval.flatten())) # changed and adapted to our problem setting
            else:
                outvector1 = np.concatenate((np.array([outeriter, f0val]), fval.flatten())) # changed and adapted to our problem setting
            outvector2 = xval.flatten()
            # Log
            logger.info("outvector1 = {}".format(outvector1))
            logger.info("outvector2 = {}\n".format(outvector2))
        # The iterations starts
        kktnorm = kkttol + 10
        outit = 0
        if self.globalize == False: # changed and adapted to our problem setting
            while (kktnorm > kkttol) and (outit < maxoutit):
                outit += 1
                outeriter += 1
                # The MMA subproblem is solved at the point xval:
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, low, upp = \
                mmasub(m, n, outeriter, xval, xmin, xmax, xold1, xold2, f0val, df0dx, fval, dfdx, low, upp, a0, a, c, d,
                       move)
                # Some vectors are updated:
                xold2 = xold1.copy()
                xold1 = xval.copy()
                xval = xmma.copy()
                # Re-calculate function values and gradients of the objective and constraints functions
                f0val, df0dx, fval, dfdx = self.problem(xval.T[0]) # changed and adapted to our problem setting
                # The residual vector of the KKT conditions is calculated
                residu, kktnorm, residumax = \
                    kktcheck(m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c, d)
                outvector1 = np.concatenate((np.array([outeriter]), xval.flatten()))
                outvector2 = np.concatenate((np.array([f0val]), fval.flatten()))
                # Log
                logger.info("outvector1 = {}".format(outvector1))
                logger.info("outvector2 = {}".format(outvector2))
                logger.info("kktnorm    = {}\n".format(kktnorm))
        else:
            raa0 = 0.01
            raa = 0.01 * eeem
            raa0eps = 0.000001
            raaeps = 0.000001 * eeem
            while (kktnorm > kkttol) and (outit < maxoutit):
                outit += 1
                outeriter += 1
                # The parameters low, upp, raa0 and raa are calculated:
                low, upp, raa0, raa = \
                    asymp(outeriter, n, xval, xold1, xold2, xmin, xmax, low, upp, raa0, raa, raa0eps, raaeps, df0dx,
                          dfdx)
                # The MMA subproblem is solved at the point xval:
                xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = \
                    gcmmasub(m, n, iter, epsimin, xval, xmin, xmax, low, upp, raa0, raa, f0val, df0dx, fval, dfdx, a0,
                             a, c, d)
                # The user should now calculate function values (no gradients) of the objective- and constraint
                # functions at the point xmma ( = the optimal solution of the subproblem).
                f0valnew, fvalnew = self.problem(xmma, no_grad=True) # changed and adapted to our problem setting
                # It is checked if the approximations are conservative:
                conserv = concheck(m, epsimin, f0app, f0valnew, fapp, fvalnew)
                # While the approximations are non-conservative (conserv=0), repeated inner iterations are made:
                innerit = 0
                if conserv == 0:
                    while conserv == 0 and innerit <= 15:
                        innerit += 1
                        # New values on the parameters raa0 and raa are calculated:
                        raa0, raa = raaupdate(xmma, xval, xmin, xmax, low, upp, f0valnew, fvalnew, f0app, fapp, raa0, \
                                              raa, raa0eps, raaeps, epsimin)
                        # The GCMMA subproblem is solved with these new raa0 and raa:
                        xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, f0app, fapp = gcmmasub(m, n, iter, epsimin, xval,
                                                                                            xmin, \
                                                                                            xmax, low, upp, raa0, raa,
                                                                                            f0val, df0dx, fval, dfdx,
                                                                                            a0, a, c, d)
                        # The user should now calculate function values (no gradients) of the objective- and
                        # constraint functions at the point xmma ( = the optimal solution of the subproblem).
                        f0valnew, fvalnew = self.problem(xmma, no_grad=True) # changed and adapted to our problem setting
                        # It is checked if the approximations have become conservative:
                        conserv = concheck(m, epsimin, f0app, f0valnew, fapp, fvalnew)
                # Some vectors are updated:
                xold2 = xold1.copy()
                xold1 = xval.copy()
                xval = xmma.copy()
                # Re-calculate function values and gradients of the objective and constraints functions
                f0val, df0dx, fval, dfdx = self.problem(xmma) # changed and adapted to our problem setting
                # The residual vector of the KKT conditions is calculated
                residu, kktnorm, residumax = \
                    kktcheck(m, n, xmma, ymma, zmma, lam, xsi, eta, mu, zet, s, xmin, xmax, df0dx, fval, dfdx, a0, a, c,
                             d)
                if self.globalize: # changed and adapted to our problem setting
                    outvector1 = np.concatenate((np.array([outeriter, innerit, f0val]), fval.flatten()))
                else: # changed and adapted to our problem setting
                    outvector1 = np.concatenate((np.array([outeriter, f0val]), fval.flatten()))
                outvector2 = xval.flatten()
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