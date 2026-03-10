import numpy as np
import matplotlib.pyplot as plt
from Class_FCI import FCI

Nx=3
Ny=3
Nt=10
dx=np.ones(Nx)*2
dy=np.ones(Ny)*3
dt=4
eps=np.ones(Nx*Ny)
mu=np.ones(Nx*Ny)

solver=FCI(Nx,Ny,Nt,dx,dy,dt,eps,mu)
solver.construct_update_matrix()