import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee

L = 1
Nx = 300
Ny = 300
c = 1
CFL = 0.9
dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
J0 = 10
Wc = 0.2/dt
width = 2/Wc
tc = 5*width
Nt = int(20*tc/dt)

xs = Nx//4
ys = Ny//3
xr = Nx//2
yr = Ny//2

N_PML = 30
m = 4

###
# Without PML
###
# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,m,PML=False)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)

# solver.animate(speed = 1)
# solver.restart()

# solver.update_loop()
# solver.show_recorder()
# solver.restart()

###
# With PML
###
solver = Yee(L,Nx,Ny,Nt,dt,N_PML,m,PML=True)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)

# solver.show_PML() # PML profile

solver.animate(speed = 10)
solver.restart()

solver.update_loop()
solver.show_recorder()
solver.restart()