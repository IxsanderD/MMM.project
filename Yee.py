import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee

L = 5
Nx = 100
Ny = 100
c = 1
CFL = 1
dt = CFL/c/np.sqrt(1/np.min(L/Nx)**2+1/np.min(L/Ny)**2)
xs = Nx//2
ys = Ny//2
J0 = 1
width = 1
tc = 6*width
Wc = 1
Nt = int(L*Nx/c+tc/dt)

solver = Yee(L,Nx,Ny,Nt,dt)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.update_loop()

plt.imshow(solver.Ez)
plt.colorbar()
plt.show()