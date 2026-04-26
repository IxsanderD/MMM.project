import numpy as np
from Class_Yee import Yee
from Class_FCI import FCI
from scipy.constants import c

# In this document, we will analyse the ability of both the Yee and FCI scheme to reproduce the analytical solution of a 2D-TM EM wave

# We first start by analysing propagation in the x-direction. We fill in the parameters for the Yee-scheme

Nx = 401
Ny = 401
L = 1
CFL = 0.90
dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
J0 = 50
Wc = 0.15/dt
width = 3/Wc
tc = 5*width
Nt = int(5*tc/dt)
N_PML=30

# xs=Nx//4
# ys=Ny//2
# xr=xs+2
# yr=Ny//2

# solver_Yee=Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,m=4)
# solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
# solver_Yee.add_recorder(xr,yr)
# # solver_Yee.animate(speed=10)

# solver_Yee.update_loop()
# solver_Yee.show_recorder()

# solver_Yee.analytical_solution(plot_all=False,frequency_limit=Wc+3/width)

# Now we do the same for the FCI scheme

Nx=301
Ny=301
L = 1
dx=np.ones(Nx)*L/Nx
dy=np.ones(Ny)*L/Ny
Nt=400
J0 = 10
Wc = 0.05/dt
width = 3/Wc
tc = 5*width
tf = 40*tc
dt = tf/Nt

xs=Nx//2
ys=Ny//2
xr=3*Nx//5
yr=Ny//2

solver_FCI=FCI(Nt,dx,dy,dt,k_max=1,sigma_max=10,drude=False)
solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
solver_FCI.add_recorder(xr,yr)
solver_FCI.construct_matrices()
solver_FCI.animate(speed=3)

solver_FCI.update_loop()
solver_FCI.analytical_sol(p_all=False,f_lim=Wc+3/width)

# Now we do the same analysis, but now in the diagonal propagation direction. We again fill in the parameters for Yee

# Nx=401
# Ny=401
# L = 1
# CFL = 0.90
# dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
# J0 = 10
# Wc = 0.15/dt
# width = 3/Wc
# tc = 5*width
# Nt = int(5*tc/dt)
# N_PML=70

# xs=Nx//2
# ys=Ny//2
# xr=xs+1
# yr=xs+1

# solver_Yee=Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,m=4)
# solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
# solver_Yee.add_recorder(xr,yr)
# # solver_Yee.animate(speed=10)
# solver_Yee.update_loop()
# solver_Yee.show_recorder()
# solver_Yee.analytical_solution(plot_all=False,frequency_limit=Wc+3/width)

# And lastly also for FCI

# Nx = 301
# Ny = 301
# L = 1
# dx = np.ones(Nx)*L/Nx
# dy = np.ones(Ny)*L/Ny
# Nt = 400
# J0 = 10
# Wc = 0.05/dt
# width = 3/Wc
# tc = 5*width
# tf = 30*tc
# dt = tf/Nt

# xs=Nx//2
# ys=Ny//2
# xr=3*Nx//5
# yr=3*Ny//5

# solver_FCI=FCI(Nt,dx,dy,dt,k_max=3.221,sigma_max=10,drude=False)
# solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
# solver_FCI.add_recorder(xr,yr)
# solver_FCI.construct_matrices()
# # solver_FCI.animate(speed=3)
# solver_FCI.update_loop()
# solver_FCI.show_recorder()
# solver_FCI.analytical_sol(p_all=False,f_lim=Wc+3/width)

