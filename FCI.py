import numpy as np
import matplotlib.pyplot as plt
from Class_FCI import FCI

Nx=50
Ny=50
Nt=50
dx=np.ones(Nx)
dy=np.ones(Ny)
dt=1
c=1
eps=np.ones(Nx*Ny)
mu=np.ones(Nx*Ny)
J0 = 10
width = np.sum(dx)/c
tc = 5*width
Wc = 2*np.pi*0.5/tc

xs = Nx//4
ys = Ny//2

solver=FCI(Nx,Ny,Nt,dx,dy,dt,eps,mu)
solver.construct_update_matrix()
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.animate()
solver.restart()
