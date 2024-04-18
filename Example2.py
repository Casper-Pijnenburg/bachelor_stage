from lib2to3.pgen2.token import N_TOKENS
import numpy as np
import matplotlib.pyplot as plt
import solver

'''
Example of how to use the nested grid method.
System is a dielectric slab with approximated boundary conditions
'''

def eps(x, y, z, eps1, eps2, d):
    N = np.size(x)
    d_arg_pos = np.argmin(np.abs(z - d/2))
    d_arg_neg = np.argmin(np.abs(z + d/2))

    eps = np.ones((N, N, N)) * eps2
    eps[:, :, d_arg_neg:d_arg_pos] = eps1

    return eps

N = 100 # No. of grid points used in each grid
n = 10 # No. of nested grids

# Define the extents. Extents are ranging from 10^0 to 10^6 using 15 equally spaced powers
powers = np.linspace(0, 6, n)
extents = 10 ** powers

# Store Grid classes in a list
grids = []
for ext in extents:
    grids.append(solver.Grid(N = N, extent = ext, base = 'lin', tol = 1e-9, center = (0, 0, 0)))

u = solver.NestedGridSolve(eps, (5, 1, 2/3), grids[::-1])


# Calculate the exact solution for comparison
x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
z = np.linspace(-1, 1, N)
eps, E = solver.DielectricSlab(x, y, z, d = 2/3, eps1 = 5, eps2 = 1, q = -1)

# Caculate the M.R.E and plot the results
u_err = np.copy(u)
E_err = np.copy(E)
u_err[np.isinf(u_err)] = 1
E_err[np.isinf(E_err)] = 1

MRE = np.mean(np.abs(u_err[1:-1, 1:-1, 1:-1] - E_err[1:-1, 1:-1, 1:-1]) / E_err[1:-1, 1:-1, 1:-1])

fig = plt.figure(figsize = (8, 8))

plt.plot(z, E[np.argmin(np.abs(x)), np.argmin(np.abs(y)), :], label = "Exact", color = 'black', zorder = 0)
plt.scatter(z, u[np.argmin(np.abs(x)), np.argmin(np.abs(y)), :], label = "Numerical", color = 'red', zorder = 1, s = 20)

plt.legend()
plt.title('Dielectric slab with a M.R.E of {:.2e}'.format(MRE))

plt.show()