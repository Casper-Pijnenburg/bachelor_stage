"""
Poisson equation solver for non-uniform dielectrics

Written by Casper Pijnenburg as part of a bachelor internship
at the department of theory of condensed matter at Radboud university

"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab, norm, inv
from scipy.interpolate import interp2d
from scipy.interpolate import RegularGridInterpolator
import time
import logging

format = '%(levelname)s at time: %(asctime)s in function: %(funcName)s - %(message)s'

logging.basicConfig(level = logging.INFO, format = format, datefmt = '%H:%M:%S')

def construct_boundry_vector(u, eps, x, y, z):
    """ Returns a vector containing the boundary terms

        Parameters 
        ----------
        u : N by N by N array
            u contains the known boundary terms and zeros everywhere else
        eps : N by N by N array
            eps values at each point in a 3D grid
        x, y, z : Length N arrays
            x, y and z points of the grid

        Returns
        ----------
        b : Length (N - 2)^3 array
            b contains the boundary terms with the correct α factor reshaped to length (N - 2)^3
    """

    if not (np.ndim(u) == 3 and np.ndim(eps) == 3):
        logging.critical("u and eps need to be 3D arrays")
        exit()
    
    if not (np.shape(u) == np.shape(eps)):
        logging.critical("Shapes of u and eps don't match")
        exit()

    if not (np.size(x) == np.size(y) and np.size(y) == np.size(z)):
        logging.critical("x, y and z arrays must have the same length")
        exit()


    # dx, dy and dz from x, y and z
    dx = (x[1:] - x[:-1])
    dy = (y[1:] - y[:-1])
    dz = (z[1:] - z[:-1])

    # calculating α1 to α6 from eps and dx, dy and dz
    a1 = (eps[1:-1,1:-1,1:-1] * dy[None, 1:, None] * dz[None, None, 1:] + eps[1:-1,:-2,1:-1] * dy[None, :-1, None] * dz[None, None, 1:]
            + eps[1:-1,1:-1,:-2] * dy[None, 1:, None] * dz[None, None, :-1] + eps[1:-1,:-2,:-2] * dy[None, :-1, None] * dz[None, None, :-1]) / (4 * dx[1:, None, None])
    a2 = (eps[:-2, 1:-1, 1:-1] * dy[None, 1:, None] * dz[None, None, 1:] + eps[:-2,:-2,1:-1] * dy[None, :-1, None] * dz[None, None, 1:]
                + eps[:-2, 1:-1, :-2] * dy[None, 1:, None] * dz[None, None, :-1] + eps[:-2, :-2, :-2] * dy[None, :-1, None] * dz[None, None, :-1]) / (4 * dx[:-1, None, None])
    a3 = (eps[1:-1, 1:-1, 1:-1] * dx[1:, None, None] * dz[None, None, 1:] + eps[:-2, 1:-1, 1:-1] * dx[:-1, None, None] * dz[None, None, 1:]
                + eps[1:-1, 1:-1, :-2] * dx[1:, None, None] * dz[None, None, :-1] + eps[:-2, 1:-1, :-2] * dx[:-1, None, None] * dz[None, None, :-1]) / (4 * dy[None, 1:, None])
    a4 = (eps[1:-1, :-2, 1:-1] * dx[1:, None, None] * dz[None, None, 1:] + eps[:-2, :-2, 1:-1] * dx[:-1, None, None] * dz[None, None, 1:]
                + eps[1:-1, :-2, :-2] * dx[1:, None, None] * dz[None, None, :-1] + eps[:-2, :-2, :-2] * dx[:-1, None, None] * dz[None, None, :-1]) / (4 * dy[None, :-1, None])
    a5 = (eps[1:-1, 1:-1, 1:-1] * dx[1:, None, None] * dy[None, 1:, None] + eps[:-2, 1:-1, 1:-1] * dx[:-1, None, None] * dy[None, 1:, None]
                + eps[1:-1, :-2, 1:-1] * dx[1:, None, None] * dy[None, :-1, None] + eps[:-2, :-2, 1:-1] * dx[:-1, None, None] * dy[None, :-1, None]) / (4 * dz[None, None, 1:])
    a6 = (eps[1:-1, 1:-1, :-2] * dx[1:, None, None] * dy[None, 1:, None] + eps[:-2, 1:-1, :-2] * dx[:-1, None, None] * dy[None, 1:, None]
                + eps[1:-1, :-2, :-2] * dx[1:, None, None] * dy[None, :-1, None] + eps[:-2, :-2, :-2] * dx[:-1, None, None] * dy[None, :-1, None]) / (4 * dz[None, None, :-1])


    # array containing boundary values with corresponding α factor
    b = -(a1 * u[2:, 1:-1, 1:-1] + a2 * u[:-2, 1:-1, 1:-1] 
        + a3 * u[1:-1, 2:, 1:-1] + a4 * u[1:-1, :-2, 1:-1]
        + a5 * u[1:-1, 1:-1, 2:] + a6 * u[1:-1, 1:-1, :-2])


    return b.reshape(len(b) ** 3)


def PoissonMatrix(eps, x, y, z):
    """ Returns Poisson matrix and a preconditioner in Scipy sparse matrix in csc format.

        Parameters 
        ----------
        eps : N by N by N array
            eps values at each point in a 3D grid
        x, y, z : Length N arrays
            x, y and z points of the grid

        Returns
        ----------
        A : (N-2)^3 by (N-2)^3 SciPy sparse matrix in csc format
            A contains the α factors for each gridpoint on the diagonals corresponding with the derivatives and boundaries of the grid
        P : (N-2)^3 by (N-2)^3 SciPy sparse matrix in csc format
            Precondition matrix of A
    """
    if not (np.ndim(eps) == 3):
        logging.critical("eps needs to be a 3D array")
        exit()

    if not (np.shape(eps)[0] == np.shape(eps)[1] and np.shape(eps)[1] == np.shape(eps)[0]):
        logging.critical("The eps matrix is not cubic")
        exit()

    if not (np.size(x) == np.size(y) and np.size(y) == np.size(z)):
        logging.critical("x, y and z arrays must have the same length")
        exit()

    # dx, dy and dz from x, y and z
    dx = (x[1:] - x[:-1])
    dy = (y[1:] - y[:-1])
    dz = (z[1:] - z[:-1])

    # calculating α0 to α6 from eps and dx, dy and dz
    a1 = (eps[1:-1,1:-1,1:-1] * dy[None, 1:, None] * dz[None, None, 1:] + eps[1:-1,:-2,1:-1] * dy[None, :-1, None] * dz[None, None, 1:]
            + eps[1:-1,1:-1,:-2] * dy[None, 1:, None] * dz[None, None, :-1] + eps[1:-1,:-2,:-2] * dy[None, :-1, None] * dz[None, None, :-1]) / (4 * dx[1:, None, None])
    a2 = (eps[:-2, 1:-1, 1:-1] * dy[None, 1:, None] * dz[None, None, 1:] + eps[:-2,:-2,1:-1] * dy[None, :-1, None] * dz[None, None, 1:]
                + eps[:-2, 1:-1, :-2] * dy[None, 1:, None] * dz[None, None, :-1] + eps[:-2, :-2, :-2] * dy[None, :-1, None] * dz[None, None, :-1]) / (4 * dx[:-1, None, None])
    a3 = (eps[1:-1, 1:-1, 1:-1] * dx[1:, None, None] * dz[None, None, 1:] + eps[:-2, 1:-1, 1:-1] * dx[:-1, None, None] * dz[None, None, 1:]
                + eps[1:-1, 1:-1, :-2] * dx[1:, None, None] * dz[None, None, :-1] + eps[:-2, 1:-1, :-2] * dx[:-1, None, None] * dz[None, None, :-1]) / (4 * dy[None, 1:, None])
    a4 = (eps[1:-1, :-2, 1:-1] * dx[1:, None, None] * dz[None, None, 1:] + eps[:-2, :-2, 1:-1] * dx[:-1, None, None] * dz[None, None, 1:]
                + eps[1:-1, :-2, :-2] * dx[1:, None, None] * dz[None, None, :-1] + eps[:-2, :-2, :-2] * dx[:-1, None, None] * dz[None, None, :-1]) / (4 * dy[None, :-1, None])
    a5 = (eps[1:-1, 1:-1, 1:-1] * dx[1:, None, None] * dy[None, 1:, None] + eps[:-2, 1:-1, 1:-1] * dx[:-1, None, None] * dy[None, 1:, None]
                + eps[1:-1, :-2, 1:-1] * dx[1:, None, None] * dy[None, :-1, None] + eps[:-2, :-2, 1:-1] * dx[:-1, None, None] * dy[None, :-1, None]) / (4 * dz[None, None, 1:])
    a6 = (eps[1:-1, 1:-1, :-2] * dx[1:, None, None] * dy[None, 1:, None] + eps[:-2, 1:-1, :-2] * dx[:-1, None, None] * dy[None, 1:, None]
                + eps[1:-1, :-2, :-2] * dx[1:, None, None] * dy[None, :-1, None] + eps[:-2, :-2, :-2] * dx[:-1, None, None] * dy[None, :-1, None]) / (4 * dz[None, None, :-1])

    a0 = a1 + a2 + a3 + a4 + a5 + a6

    # N is now the number of grid points without the boundries
    N = len(eps) - 2
    N2 = N ** 2
    N3 = N ** 3
    
    # reshape α factors to column vectors
    a1 = a1.reshape(N3)
    a2 = a2.reshape(N3)
    a3 = a3.reshape(N3)
    a4 = a4.reshape(N3)
    a5 = a5.reshape(N3)
    a6 = a6.reshape(N3)
    a0 = a0.reshape(N3)
    

    # Array to construct the correct z derivatives in the matrix
    Mz = np.arange(N3 - 1)
    Mz = ((Mz + 1) % N != 0)

    # Array to construct the correct y derivatives in the matrix
    My = np.arange(N3 - N)
    My = (My % N2 <= N2 - N - 1)

    # Diagonals of the matrix
    diagonals = [
        a2[N2:],
        My * a4[N:],
        Mz * a6[1:],
        -a0,
        Mz * a5[:-1],
        My * a3[:-N],
        a1[:-N2]
    ]
    
    # Create the sparse matrix from diagonals
    A = sparse.diags(diagonals, [-N2, -N, -1, 0, 1, N, N2]).tocsc()

    # Precondition matrix
    P = sparse.diags([-1/a0], [0]).tocsc()

    return A, P


def Jacobi_method(u, rho, eps, x, y, z, MAX_DIFF = 1e-5, MAX_ITER = 50000, callback = None):
    """ Solve the Poisson equation iteratively

        Parameters 
        ----------
        u : N by N by N array
            u contains the known boundary terms and zeros everywhere else
        rho : N by N by N array
            rho contains the charge density at each point in a 3D grid
        eps : N by N by N array
            eps values at each point in a 3D grid
        dx, dy, dz : Length N-1 arrays
            Stepsize of the grid at for x, y and z at each grid point
        MAX_DIFF : float
            Iterations stop if the mean difference between the last two iterations is less than MAX_DIFF.
            Default is 1e-5
        MAX_ITER : int
            Maximum number of iterations
        Callback : function
            callback(x) gets used each iteration on the current solution.
            Default is None

        Returns
        ----------
        u : N^3 size array
            Current u after this function stops iterating
        status : int
            0 if the function stops before reaching MAX_ITER.
            1 if the function reaches MAX_ITER
        iterations : int
            Number of iterations completed
    """
    if not (np.ndim(u) == 3 and np.ndim(rho) == 3 and np.ndim(eps) == 3):
        logging.critical("u, rho and eps need to be 3D arrays")
        exit()

    if not (np.shape(u) == np.shape(rho) and np.shape(rho) == np.shape(eps)):
        logging.critical("Shapes of u, rho and eps don't match")
        exit()
    
    if not (np.size(x) == np.size(y) and np.size(y) == np.size(z)):
        logging.critical("x, y and z arrays must have the same length")
        exit()

    if not (MAX_DIFF > 0):
        logging.critical("Maximum allowed difference has to be greater than zero")
        exit()   

    if not (MAX_ITER > 0):
        logging.critical("Maximum iterations has to be positive")
        exit() 

    # Copy of u to store next iterations in
    u_next = np.copy(u)

    # dx, dy and dz from x, y and z
    dx = (x[1:] - x[:-1])
    dy = (y[1:] - y[:-1])
    dz = (z[1:] - z[:-1])

    # calculating α0 to α6 from eps and dx, dy and dz
    a1 = (eps[1:-1,1:-1,1:-1] * dy[None, 1:, None] * dz[None, None, 1:] + eps[1:-1,:-2,1:-1] * dy[None, :-1, None] * dz[None, None, 1:]
            + eps[1:-1,1:-1,:-2] * dy[None, 1:, None] * dz[None, None, :-1] + eps[1:-1,:-2,:-2] * dy[None, :-1, None] * dz[None, None, :-1]) / (4 * dx[1:, None, None])
    a2 = (eps[:-2, 1:-1, 1:-1] * dy[None, 1:, None] * dz[None, None, 1:] + eps[:-2,:-2,1:-1] * dy[None, :-1, None] * dz[None, None, 1:]
                + eps[:-2, 1:-1, :-2] * dy[None, 1:, None] * dz[None, None, :-1] + eps[:-2, :-2, :-2] * dy[None, :-1, None] * dz[None, None, :-1]) / (4 * dx[:-1, None, None])
    a3 = (eps[1:-1, 1:-1, 1:-1] * dx[1:, None, None] * dz[None, None, 1:] + eps[:-2, 1:-1, 1:-1] * dx[:-1, None, None] * dz[None, None, 1:]
                + eps[1:-1, 1:-1, :-2] * dx[1:, None, None] * dz[None, None, :-1] + eps[:-2, 1:-1, :-2] * dx[:-1, None, None] * dz[None, None, :-1]) / (4 * dy[None, 1:, None])
    a4 = (eps[1:-1, :-2, 1:-1] * dx[1:, None, None] * dz[None, None, 1:] + eps[:-2, :-2, 1:-1] * dx[:-1, None, None] * dz[None, None, 1:]
                + eps[1:-1, :-2, :-2] * dx[1:, None, None] * dz[None, None, :-1] + eps[:-2, :-2, :-2] * dx[:-1, None, None] * dz[None, None, :-1]) / (4 * dy[None, :-1, None])
    a5 = (eps[1:-1, 1:-1, 1:-1] * dx[1:, None, None] * dy[None, 1:, None] + eps[:-2, 1:-1, 1:-1] * dx[:-1, None, None] * dy[None, 1:, None]
                + eps[1:-1, :-2, 1:-1] * dx[1:, None, None] * dy[None, :-1, None] + eps[:-2, :-2, 1:-1] * dx[:-1, None, None] * dy[None, :-1, None]) / (4 * dz[None, None, 1:])
    a6 = (eps[1:-1, 1:-1, :-2] * dx[1:, None, None] * dy[None, 1:, None] + eps[:-2, 1:-1, :-2] * dx[:-1, None, None] * dy[None, 1:, None]
                + eps[1:-1, :-2, :-2] * dx[1:, None, None] * dy[None, :-1, None] + eps[:-2, :-2, :-2] * dx[:-1, None, None] * dy[None, :-1, None]) / (4 * dz[None, None, :-1])

    a0 = a1 + a2 + a3 + a4 + a5 + a6


    # Charge density times volume
    rhoV = rho[1:-1, 1:-1, 1:-1] * (dx[1:, None, None] + dx[:-1, None, None]) * (dy[None, 1:, None] + dy[None, :-1, None]) * (dz[None, None, 1:] + dz[None, None, :-1]) / 8

    for i in range(int(MAX_ITER)):
        # Calculate next state from previous state
        u_next[1:-1, 1:-1, 1:-1] = (a1 * u[2:, 1:-1, 1:-1] + a2 * u[:-2, 1:-1, 1:-1] 
                                  + a3 * u[1:-1, 2:, 1:-1] + a4 * u[1:-1, :-2, 1:-1]
                                  + a5 * u[1:-1, 1:-1, 2:] + a6 * u[1:-1, 1:-1, :-2] - rhoV) / a0

        if callback is not None:
            callback(u_next)
        # Check if convergence is reached                         
        if (np.max(np.abs(u - u_next)) < MAX_DIFF):
            return u_next, 0, i

        # Store current state in previous state    
        u[:] = u_next

    return u, 1, i



def ScipySolve(A, b, x0, tol, MAX_ITER, P = None, callback = None):
    """ Returns the solution x for Ax=b using SciPy BiCGStab

        Parameters 
        ----------
        A : N by N sparse matrix
            A in Ax=b
        b : Length N array
            b in Ax=b
        x0 : Length N array
            Starting guess for solution
        tol : float
            Tolerance for convergence
        MAX_ITER : int
            Maximum number of iterations allowed
        P : N by N sparse matrix
            Preconditioner for A.
            Default is None
        Callback : function
            callback(x) gets used each iteration on the current solution.
            Default is None
        Returns
        ----------
        x : Length N array
            x is the solution found using BiCGStab
        status : int
            0 for a succesful exit.
            >0 when convergence is not reached.
            <0 for illegal input or breakdown
        iterations : int
            Number of iterations done by the BiCGStab
    """

    if not (A.get_shape()[0] == np.shape(b)[0] and np.shape(b) == np.shape(x0)):
        logging.critical("Shapes of A and b don't match")
        exit()

    if not (np.shape(A) == np.shape(P)):
        logging.critical("Shapes of A and P don't match")

    # Callback function to count iterations and do the actual callback
    iterations = 0
    def _callback(x):
        nonlocal iterations
        iterations += 1
    
        if callback is not None:
            callback(x)

    # run SciPy bigcgstab with given parameters
    x, status = bicgstab(A, b, x0 = x0, tol = tol, callback = _callback, maxiter = MAX_ITER, M = P)
    return x, status, iterations

def solve(eps, rho, u_init, x, y, z, solver = 'bicgstab', tol = 1e-5, MAX_ITER = 5e4, callback = None):
    """ Solve the Poisson equation given eps and rho

        Parameters 
        ----------
        eps : N by N by N array
            Contains the value of eps inbetween each gridpoint
        rho : N by N by N array
            Contains the charge density at each gridpoint
        u_init : N by N by N array
            Contains the boundaries and initial guess at each gridpoint
        x, y, z : Length N arrays
            Contains the x, y, z values of the grid.
            Used to find dx, dy and dz
        solver : string
            To specify which solver to use: bicgstab or jacobi.
            Default is bicgstab because it's much faster
        tol : float
            Goes into tol or MAX_DIFF argument depending on solver.
            Default is 1e-9
        MAX_ITER : int
            Maximum number of iterations for both solvers.
            Default is 5e4
        Callback : function
            callback(x) gets used each iteration on the current solution.
            Default is None.
            Care that x is a length (N-2)^3 array for the matrix method
            and x is a N by N by N array for the iterative method

        Returns
        ----------
        u : N by N by N array
            Solution to u using the specified solver
    """
    if not (np.ndim(eps) == 3 and np.ndim(rho) == 3 and np.ndim(u_init) == 3):
        logging.critical("Eps, rho and u_init should be 3D arrays")
        exit()

    if not (np.size(x) == np.size(y) and np.size(y) == np.size(z)):
        logging.critical("x, y and z arrays must have the same length")
        exit()

    if not (tol > 0):
        logging.critical("Tol has to be greater than zero")
        exit()

    if not (MAX_ITER > 0):
        logging.critical("Maximum iterations has to be positive")
        exit()


    if solver == 'bicgstab':
        logging.info("Attempting to solve using the matrix method")


        # Number of grid points in each direction
        N = len(eps)

        # dx, dy and dz from x, y and z
        dx = (x[1:] - x[:-1])
        dy = (y[1:] - y[:-1])
        dz = (z[1:] - z[:-1])

        # Extract initial guess from u_init and stores it in x0
        x0 = np.zeros((N - 2, N - 2, N - 2))
        x0[:] = u_init[1:-1, 1:-1, 1:-1]
        x0 = x0.reshape((N - 2) ** 3)

        # Construct Poisson matrix and its preconditioner
        A, P = PoissonMatrix(eps, x, y, z)
        logging.info("Matrix constructed")


        """
        norm_A = norm(P.multiply(A))
        norm_inv_A = norm(inv(P.multiply(A)))
        cond = norm_A * norm_inv_A
        logging.info("Condition number A: {}".format(cond))
        """

        # Constructs the boundary vector from u_init
        u_init[1:-1, 1:-1, 1:-1] = np.zeros((N - 2, N - 2, N - 2))
        bound_vector = construct_boundry_vector(u_init, eps, x, y, z)

        # Charge density times volume
        rhoV = rho[1:-1, 1:-1, 1:-1] * (dx[1:, None, None] + dx[:-1, None, None]) * (dy[None, 1:, None] + dy[None, :-1, None]) * (dz[None, None, 1:] + dz[None, None, :-1]) / 8

        # Calculate b from the boundary vector and the charge distribution
        b = bound_vector + rhoV.reshape((N - 2) ** 3)
        logging.info("Boundary vector contructed and started BiCGStab...")

        start = time.perf_counter()
        # Solves the system using bigstab
        x, status, iterations = ScipySolve(A, b, x0, tol, MAX_ITER, P, callback)
        end = time.perf_counter()

        if status > 0:
            u_init[1:-1, 1:-1, 1:-1] = x.reshape(N - 2, N - 2, N - 2)
            logging.warning("Maximum number of iterations reached in {:.3f} seconds".format(end - start))
            return u_init
        elif status < 0:
            logging.error("Illegal input or breakdown")
            exit()
        elif status == 0:
            u_init[1:-1, 1:-1, 1:-1] = x.reshape(N - 2, N - 2, N - 2)
            logging.info("Convergence criterium reached in {} iterations and {:.3f} seconds".format(iterations, end - start))
            return u_init
    
    elif solver == 'jacobi':
        logging.info("Attempting to solve using the iterative method...")

        start = time.perf_counter()
        # Solves the system using the iterative method
        u_init, status, iterations = Jacobi_method(u_init, rho, eps, x, y, z, tol, MAX_ITER, callback)
        end = time.perf_counter()

        if status == 0:
            logging.info("Convergence criterium reached in {} iterations and {:.3f} seconds".format(iterations, end - start))
            return u_init
        elif status == 1:
            logging.warning("Maximum number of iterations reached in {:.3f} seconds".format(end - start))
            return u_init
    else:
        logging.error("Supported solvers are 'bicgstab' or 'jacobi'")
        return u_init




def gridValue(val, x):
    """ Returns the value in x that is closest to val

        Parameters 
        ----------
        val : float
            Value to find in x
        x : Length N array
            Array of grid points

        Returns
        ----------
        x_val : float
            Closest element in x to val
    """

    arg = np.argmin(np.abs(x - val))
    x_val = x[arg]
    return x_val

def _delta(x, y, z, loc = (0, 0, 0)):
    """ 3D delta function approximation using a normalized Gaussian where
        σ is the max of the min of the three grid spaces for x, y and z

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        loc : tuple of floats
            Location of the center of the gaussian.
            Default is (0, 0, 0)
        Returns
        ----------
        delta : N by N by N array
            3D array of an approximate delta function
    """
    σx = np.min(x[1:] - x[:-1])
    σy = np.min(y[1:] - y[:-1])
    σz = np.min(z[1:] - z[:-1])

    σ = np.max([σx, σy, σz])

    x_loc, y_loc, z_loc = loc

    X, Y, Z = np.meshgrid(x, y, z)
    delta = np.exp(-((X - x_loc) ** 2 + (Y - y_loc) ** 2 + (Z - z_loc) ** 2) / (2 * σ ** 2)) / (σ ** 3 * np.sqrt(2 * np.pi) ** 3)
    return delta


def ChargeDist(q, x, y, z, loc = (0, 0, 0)):
    """ Create charge distrubution from q and location

        Parameters 
        ----------
        q : float
            Charge value
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        loc : tuple of floats
            Location of the center of the gaussian.
            Default is (0, 0, 0)
        Returns
        ----------
        rho : N by N by N array
            3D array of the charge distribution
    """
    rho = q * _delta(x, y, z, loc)
    return rho


def SingleCharge(x, y, z, q, eps, loc = (0, 0, 0)):
    """ Exact solution for the potential of a single charge

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        q : float
            Charge value
        eps : float
            Eps value
        loc : tuple of floats
            Location of the charge.
            Default is (0, 0, 0)
        Returns
        ----------
        phi : N by N by N array
            Exact solution at the grid points
    """
    x_loc, y_loc, z_loc = loc
    phi = -q / (4 * np.pi * eps * np.sqrt((x - x_loc) ** 2 + (y - y_loc) ** 2 + (z - z_loc) ** 2))
    return phi


def _ExactDielectricZg0(x, y, z, q, eps1, eps2, d):
    """ Used for the function _HalfSpaceSolution
    """
    s2 = x ** 2 + y ** 2
    return -(q / np.sqrt(s2 + (d - z) ** 2) + (eps1 - eps2) / (eps1 + eps2) * q / np.sqrt(s2 + (d+z)**2)) / (4 * np.pi * eps1)

def _ExactDielectricZleq0(x, y, z, q, eps1, eps2, d):
    """ Used for the function _HalfSpaceSolution
    """
    s2 = x ** 2 + y ** 2
    return -2 * eps2 / (eps1 + eps2) * q / np.sqrt(s2 + (d - z) ** 2) / (4 * np.pi * eps2)


def _HalfSpaceSolution(x, y, z, q, eps1, eps2, d):
    """ Exact solution for the potential of a single charge at z = d above the xy-plane
        with eps1 for z > 0 and eps2 for z < 0

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        q : float
            Charge value
        eps1 : float
            eps1 value
        eps2 : float
            eps2 value
        d : float
            Distance between the charge and the xy-plane
        Returns
        ----------
        phi : N by N by N array
            Exact solution at the grid points
    """

    X, Y, Z = np.meshgrid(x, y, z[z > 0])
    phi_above = _ExactDielectricZg0(X, Y, Z, q, eps1, eps2, d)
    X, Y, Z = np.meshgrid(x, y, z[z <= 0])
    phi_below = _ExactDielectricZleq0(X, Y, Z, q, eps1, eps2, d)
    phi = np.dstack((phi_below, phi_above))
    return phi


def _DielectricSlabSolution(x, y, z, q, eps1, eps2, d, no_of_image_charges = 100):
    """ Exact solution for the potential of a single charge at the origin within a dielectric slab
        of eps1 and thickness d. Outside the slab is eps2

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        q : float
            Charge value
        eps1 : float
            Eps value inside the slab
        eps2 : float
            Eps value outside the slab
        d : float
            Thickness of the slab
        no_of_image_charges : int
            Number of image charges used.
            Default is 100
        Returns
        ----------
        phi : N by N by N array
            Exact solution at the grid points
    """

    eta1 = 2 * eps1 / (eps1 + eps2)
    eta2 = (eps1 - eps2) / (eps1 + eps2)

    X, Y, Z = np.meshgrid(x, y, z[z < -d / 2])

    phi_left = np.zeros_like(Z)
    for n in range(1, no_of_image_charges + 1):
        phi_left += SingleCharge(X, Y, Z, eta1 * (eta2 ** (n - 1)) * q, eps1, (0, 0, (n - 1) * d))

    X, Y, Z = np.meshgrid(x, y, z[z > d / 2])
    phi_right = np.zeros_like(Z)
    for n in range(1, no_of_image_charges + 1):
        phi_right += SingleCharge(X, Y, Z, eta1 * (eta2 ** (n - 1)) * q, eps1, (0, 0,-(n - 1) * d))

    X, Y, Z = np.meshgrid(x, y, z[np.logical_and(z >= -d/2, z <= d/2)])
    phi_mid = SingleCharge(X, Y, Z, q, eps1)
    for n in range(1, no_of_image_charges + 1):
        phi_mid += SingleCharge(X, Y, Z, eta2 ** n * q, eps1, (0, 0, -n * d)) + SingleCharge(X, Y, Z, eta2 ** n * q, eps1, (0, 0, n * d))

    phi = np.dstack((np.dstack((phi_left, phi_mid)), phi_right))
    return phi



def UnifromDielectric(x, y, z, eps0, q, loc = (0, 0, 0)):
    """ Constructs the exact solution and eps array for a single charge in a uniform medium

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        eps0 : float
            Eps value everywhere
        q : float
            Charge value
        loc : tuple of floats
            Location of the charge.
            Default is (0, 0, 0)

        Returns
        ----------
        eps : N by N by N array
            Array of eps value at each grid point
        phi : N by N by N array
            Array of exact solution at each grid point
    """

    N = np.size(x)
    eps = np.ones((N, N, N)) * eps0
    X, Y, Z = np.meshgrid(x, y, z)
    phi = SingleCharge(X, Y, Z, q, eps0, loc)
    return eps, phi


def DielectricHalfSpace(x, y, z, d, eps1, eps2, q):
    """ Constructs the exact solution and eps array for a single charge at z = d in a dielectric halfspace
        with eps1 for z > 0 and eps2 for z < 0

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        d : float
            z location of the charge
        eps1 : float
            Eps value for z > 0
        eps2 : float
            Eps value for z < 0
        q : float
            Charge value


        Returns
        ----------
        eps : N by N by N array
            Array of eps value at each grid point
        phi : N by N by N array
            Array of exact solution at each grid point
    """
    N = np.size(x)
    eps = np.ones((N, N, N)) * eps1
    arg_z_eq_0 =  np.argmin(np.abs(z))
    zero = z[arg_z_eq_0]
    eps[:, :, :arg_z_eq_0] = eps2

    phi = _HalfSpaceSolution(x, y, z - zero, q, eps1, eps2, d - zero)

    return eps, phi

def DielectricSlab(x, y, z, d, eps1, eps2, q):
    """ Constructs the exact solution and eps array for a single charge at z = 0 in a dielectric slab
         of thickness d with eps1 inside and eps2 outside

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z are the gridpoints
        d : float
            z location of the charge
        eps1 : float
            Eps value for z > 0
        eps2 : float
            Eps value for z < 0
        q : float
            Charge value

        Returns
        ----------
        eps : N by N by N array
            Array of eps value at each grid point
        phi : N by N by N array
            Array of exact solution at each grid point
    """
    N = np.size(x)
    d_arg_pos = np.argmin(np.abs(z - d/2))
    d_arg_neg = np.argmin(np.abs(z + d/2))

    eps = np.ones((N, N, N)) * eps2
    eps[:, :, d_arg_neg:d_arg_pos] = eps1

    d = z[d_arg_pos] - z[d_arg_neg]
    phi = _DielectricSlabSolution(x, y, z, q, eps1, eps2, d, no_of_image_charges = 100)

    return eps, phi


def logArray(x0, extent, N, base = 0.1):
    """ Constructs logspaced arrays around x0 with a given extent

        Parameters 
        ----------
        x0 : float
            Location to center the array around
        extent : float
            Distance to go on both directions of x0
        N : int
            Number of points used
        base : float
            Base used in logspace.
            1 > base for points concentrated in the center.
            1 < base for points with more even spacing.
            Default is 0.25

        Returns
        ----------
        x : Length N array
            Array with logspaced points going from x0 - extent to x0 + extent
    """
    if (N % 2 != 0):
        x = np.logspace(1, np.log(extent + base) / np.log(base), np.ceil(N/2).astype(int), base = base) - base + x0
        y = ((-x)[::-1])[:-1] + 2 * x0
        return np.hstack((y, x))
    if (N % 2 == 0):
        x = np.logspace(1, np.log(extent + base) / np.log(base), np.ceil(N/2).astype(int), base = base) - base + x0
        dx = x[1] - x[0]
        x += dx / 2
        y = (-x)[::-1] + 2 * x0
        return np.hstack((y, x))


def interpolate_plane(plane, side, u, x, y, z, x_new, y_new, z_new):
    """ Interpolates a plane of u

        Parameters 
        ----------
        plane : string
            'xy', 'xz' or 'yz' for the corresponding plane
        side : int
            -1 for the plane at -x, -y or -z
            1 for the plane at x, y or z
        u : N by N by N array
            Original data
        x, y, z : Length N arrays
            x, y and z arrays on which the original data is defined
        x_new, y_new, z_new : Length N arrays
            x, y and z arrays on which to interpolate

        Returns
        ----------
        u_plane_interpolated : N by N array
            Interpolation of the plane from u on the new grid points
    """
    x_min, x_max = x_new[0], x_new[-1]
    y_min, y_max = y_new[0], y_new[-1]
    z_min, z_max = z_new[0], z_new[-1]

    x_min_arg, x_max_arg = np.argmin(np.abs(x - x_min)), np.argmin(np.abs(x - x_max))
    y_min_arg, y_max_arg = np.argmin(np.abs(y - y_min)), np.argmin(np.abs(y - y_max))
    z_min_arg, z_max_arg = np.argmin(np.abs(z - z_min)), np.argmin(np.abs(z - z_max))

    if plane == 'xy':
        if side == -1:
            u_plane = u[:, :, z_min_arg]
        elif side == 1:
            u_plane = u[:, :, z_max_arg]

        f = interp2d(x, y, u_plane, kind = 'cubic')

        u_plane_interpolated = f(x_new, y_new)

        return u_plane_interpolated
    if plane == 'xz':
        if side == -1:
            u_plane = u[:, y_min_arg, :]
        elif side == 1:
            u_plane = u[:, y_max_arg, :]
               
        f = interp2d(x, z, u_plane, kind = 'cubic')

        u_plane_interpolated = f(x_new, z_new)

        return u_plane_interpolated
    if plane == 'yz':
        if side == -1:
            u_plane = u[x_min_arg, :, :]
        elif side == 1:
            u_plane = u[x_max_arg, :, :]
               
        f = interp2d(y, z, u_plane, kind = 'cubic')

        u_plane_interpolated = f(y_new, z_new)

        return u_plane_interpolated


def interpolate_boundaries(u, x, y, z, x_new, y_new, z_new):
    """ Interpolates the boundaries from a solution on a different grid
        into a new grid

        Parameters 
        ----------
        u : N by N by N array
            Data to interpolate from
        x, y, z : Length N arrays
            x, y and z arrays on which the original data is defined
        x_new, y_new, z_new : Length N arrays
            x, y and z arrays on which to interpolate

        Returns
        ----------
        u_init : N by N by N array
            Array with interpolated boundaries
    """
    N = np.size(x_new)
    interp_boundries = np.zeros((N, N, N))

    interp_boundries[:, :, 0] = interpolate_plane('xy', -1, u, x, y, z, x_new, y_new, z_new)
    interp_boundries[:, :, -1] = interpolate_plane('xy', 1, u, x, y, z, x_new, y_new, z_new)

    interp_boundries[:, 0, :] = interpolate_plane('xz', -1, u, x, y, z, x_new, y_new, z_new)
    interp_boundries[:, -1, :] = interpolate_plane('xz', 1, u, x, y, z, x_new, y_new, z_new)

    interp_boundries[0, :, :] = interpolate_plane('yz', -1, u, x, y, z, x_new, y_new, z_new)
    interp_boundries[-1, :, :] = interpolate_plane('yz', 1, u, x, y, z, x_new, y_new, z_new)

    return interp_boundries

def create_eps(x, y, z, eps0, eps1, eps2, eps3, d):
    """ Creates eps array for the setup we're intersted in

        Parameters 
        ----------
        x, y, z : Length N arrays
            x, y and z arrays of grid points
        eps0 : float
            Eps value for z > d/2
        eps1 : float
            Eps value for -d/2 < z > d
        eps2 : float
            Eps value for -d/2 < z and x < 0
        eps2 : float
            Eps value for -d/2 < z and x > 0
        d : float
            Thickness of the slab

        Returns
        ----------
        eps : N by N by N array
            Eps array filled like the system we're interested in
    """
    N = np.size(x)
    eps = np.ones((N, N, N)) * eps0
    arg_x_0 = np.argmin(np.abs(x))
    arg_z_d_pos = np.argmin(np.abs(z - d / 2))
    arg_z_d_neg = np.argmin(np.abs(z + d / 2))


    eps[:, :, arg_z_d_neg:arg_z_d_pos] = eps1
    eps[:arg_x_0, :, :arg_z_d_neg] = eps2
    eps[arg_x_0:, :, :arg_z_d_neg] = eps3
    return eps


class Grid:
    def __init__(self, N, extent, base, q = -1, tol = 1e-6, center = (0, 0, 0)):
        self.N = N
        self.extent = extent
        self.base = base
        self.q = q
        self.tol = tol
        self.center = center


        if self.base == 'lin':
            self.x = np.linspace(-self.extent, self.extent, self.N) + self.center[0]
            self.y = np.linspace(-self.extent, self.extent, self.N) + self.center[1]
            self.z = np.linspace(-self.extent, self.extent, self.N) + self.center[2]
        else:
            self.x = logArray(self.center[0], self.extent, self.N, self.base)
            self.y = logArray(self.center[1], self.extent, self.N, self.base)
            self.z = logArray(self.center[2], self.extent, self.N, self.base)



def NestedGridSolve(eps, eps_args, grids):
    """ Solved the system using nested grids

        Parameters 
        ----------
        eps : function that returns N by N by N array
            eps(x, y, z, args) returns the the N by N by N array
            with the eps value at each point in the grid
        eps_args : tuple
            Rest of the arguments for eps() besides x, y and z
        grids : array of Grid objects
            Specifies all the grid to use
        fname_u : float
            Eps value for -d/2 < z and x < 0
        Returns
        ----------
        u : N by N by N array
            Solution on the final grid
    """
    depth = np.size(grids)

    for current_depth, grid in enumerate(grids):

        logging.info("Computing depth {} out of {}.".format(current_depth + 1, depth))
        
        eps_matrix = eps(grid.x, grid.y, grid.z, *eps_args)
        rho_matrix = ChargeDist(grid.q, grid.x, grid.y, grid.z, grid.center)

        if current_depth == 0:
            u_init = np.zeros((grid.N, grid.N, grid.N))

            u = solve(eps_matrix, rho_matrix, u_init, grid.x, grid.y, grid.z, 'bicgstab', grid.tol)

            x_old = np.copy(grid.x)
            y_old = np.copy(grid.y)
            z_old = np.copy(grid.z)

        else:
            u_interp = RegularGridInterpolator((x_old, y_old, z_old), u, method = 'linear')
            X, Y, Z = np.meshgrid(grid.x, grid.y, grid.z)

            u_init = u_interp((X, Y, Z))

            u = solve(eps_matrix, rho_matrix, u_init, grid.x, grid.y, grid.z, 'bicgstab', grid.tol)

            x_old = np.copy(grid.x)
            y_old = np.copy(grid.y)
            z_old = np.copy(grid.z)

    return u
