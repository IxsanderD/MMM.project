import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee

L = 1
Nx = 200
Ny = 100
c = 1
CFL = 0.9
dt = CFL/c/np.sqrt(1/np.min(L/Nx)**2+1/np.min(L/Ny)**2)
J0 = 10
width = 5*L/c
tc = 5*width
Wc = 2*np.pi*0.5/tc
Nt = int(7*tc/dt)

xs = Nx//4
ys = Ny//2
xr = Nx//2
yr = Ny//2

###
# Without PML
###

solver = Yee(L,Nx,Ny,Nt,dt)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)
solver.animate(speed = 100)
solver.restart()

solver.update_loop()
solver.show_recorder()
solver.restart()