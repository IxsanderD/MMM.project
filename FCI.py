import numpy as np
from scipy.constants import c
from Class_FCI import FCI
import time

start=time.perf_counter()
L = 100*10**3
Nx=61
Ny=61
Nt=100
dx=np.ones(Nx)*L/Nx
dy=np.ones(Ny)*L/Ny
c0=c
J0 = 5*10**(-7) # A/m^2
width = np.sum(dx)/(10*c0)
tc = 5*width
tf= 5*tc
dt= tf/Nt
Wc = 5/width
# Wc = 2*np.pi*c0/(7*dx[0])
# width = 5/Wc
# tc = 5*width
# dt = 5*tc/Nt

# m = 4
# k_max=1
# sigma_max= 100 #(m+1)/(150*np.pi*dx[0])

m=4
k_max=1
sigma_max=0.001 #(m+1)/(150*np.pi*dx[0])

xs = Nx//2
ys = Ny//2

solver=FCI(Nt,dx,dy,dt,k_max,sigma_max,drude=False)
# solver.add_material(3*Nx//5,4*Nx//5,3*Ny//5,4*Ny//5,1,1,1000)
solver.construct_matrices()
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xs+Nx//4,ys)
# solver.update_loop()
# solver.show_recorder()
end=time.perf_counter()
print(f"Runtime: {end - start:.6f} seconds")
solver.animate()

###
# Analytical verification
###

# xs = Nx//4
# ys = 3*Ny//4
# xr = Nx//4
# yr = Ny//4

# solver=FCI(Nt,dx,dy,dt,k_max,sigma_max)
# solver.construct_update_matrix()
# end=time.perf_counter()
# print(f"Runtime: {end - start:.6f} seconds")
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)

# solver.update_loop()
# solver.analytical_sol(plot_all=True, frequency_limit = None)