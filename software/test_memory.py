import numpy as np
import matplotlib.pyplot as plt

Ny = np.array([10,95,20,2,2,9,2,15,13,8,2,2])

Nx = 30
n = Nx**2+Nx+1
gauss = 2*n*(n+Ny)
cholesky = n/2*(n+2*Ny)+n/2

print(gauss)
print(cholesky)
print(gauss/cholesky)

Nx = 10
Ny = np.linspace(100)
n = Nx**2+Nx+1
gauss = 2*n*(n+Ny)
cholesky = n/2*(n+2*Ny)+n/2