from lib2to3.pgen2.token import N_TOKENS
import numpy as np
import matplotlib.pyplot as plt
import solver

'''
Example of how to use the solver.
System is a dielectric slab and the exact boundary conditions are used.
'''


# Define grid
N = 100

x = np.linspace(-1, 1, N)
y = np.linspace(-1, 1, N)
z = np.linspace(-1, 1, N)

X, Y, Z = np.meshgrid(x, y, z)


# Create the eps matrix and exact solution
eps, E = solver.DielectricSlab(x, y, z, d = 2/3, eps1 = 5, eps2 = 1, q = -1)

# Create charge distribution
rho = solver.ChargeDist(-1, x, y, z)

# Definite initial solution as all zeros but the exact values on the boundaries
u_init = np.copy(E)
u_init[1:-1, 1:-1, 1:-1] = np.zeros((N - 2, N - 2, N - 2))

# Solve the system using solve
u = solver.solve(eps, rho, u_init, x, y, z, solver = 'bicgstab', tol = 1e-7, MAX_ITER = 200)

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
