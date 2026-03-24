import numpy as np
import matplotlib.pyplot as plt
from Class_FCI import FCI
import time

start=time.perf_counter()

Nx=80
Ny=80
Nt=100
dx=np.ones(Nx)
dy=np.ones(Ny)
c=1
eps=np.ones(Nx*Ny)
mu=np.ones(Nx*Ny)
J0 = 20
width = np.sum(dx)/(10*c)
tc = 5*width
Wc = 2*np.pi*0.5/tc
tf=5*tc
dt=tf/Nt

xs = Nx//4
ys = Ny//2

solver=FCI(Nx,Ny,Nt,dx,dy,dt,eps,mu)
solver.construct_update_matrix()
end=time.perf_counter()
print(f"Runtime: {end - start:.6f} seconds")
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xs,ys)
# solver.update_loop()
# solver.show_recorder()
solver.animate()
