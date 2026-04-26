import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee
from Class_FCI import FCI
from scipy.constants import c

# In this document, we will analyse the ability of both the Yee and FCI scheme to reproduce the analytical solution of a 2D-TM EM wave.
# We start by defining all the parameters of our discretisation and source.

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

xs=Nx//4
ys=Ny//2

# We start by looking at vertical propagation (x-direction)

# xr=3*Nx//5
# yr=Ny//2

# We then create a Yee and FCI solver for this geometry and source and calculate the frequency repsonse of both 

# solver_Yee=Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,m=4)
# solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
# solver_Yee.add_recorder(xr,yr)
# # solver_Yee.animate(speed=10)

# solver_Yee.update_loop()
# # solver_Yee.show_recorder()
# w,H_num_Yee,H_ana=solver_Yee.analytical_solution(plot_all=False,frequency_limit=Wc+3/width)

# Now we do the same for the FCI scheme

# dx=np.ones(Nx)*L/Nx
# dy=np.ones(Ny)*L/Ny

# solver_FCI=FCI(Nt,dx,dy,dt,k_max=1,sigma_max=10,drude=False)
# solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
# solver_FCI.add_recorder(xr,yr)
# solver_FCI.construct_matrices()
# # solver_FCI.animate(speed=3)

# solver_FCI.update_loop()
# # solver_FCI.show_recorder()
# _,H_num_FCI,_=solver_FCI.analytical_sol(p_all=False,f_lim=Wc+3/width)

# We now show them together on the same plot

# plt.plot(w,H_ana,color='orange',label='Analytical')
# plt.plot(w,H_num_Yee,'x',color='blue',label='Yee')
# plt.plot(w,H_num_FCI,'+',color='red',label='FCI')
# plt.xlabel(r'Angular frequency $\omega$ [$\frac{rad}{s}$]')
# plt.ylabel(r'Transfer function H($\omega$)')
# plt.grid()
# plt.legend()
# plt.show()

# Now we do the same analysis, but in the diagonal propagation direction. We again fill in the parameters for Yee and FCI solvers

# xr=3*Nx//5
# yr=3*Nx//5

# solver_Yee=Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,m=4)
# solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
# solver_Yee.add_recorder(xr,yr)
# # solver_Yee.animate()

# solver_Yee.update_loop()
# # solver_Yee.show_recorder()
# w,H_num_Yee,H_ana=solver_Yee.analytical_solution(plot_all=False,frequency_limit=Wc+4/width)

# dx=np.ones(Nx)*L/Nx
# dy=np.ones(Ny)*L/Ny

# solver_FCI=FCI(Nt,dx,dy,dt,k_max=1,sigma_max=10,drude=False)
# solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
# solver_FCI.add_recorder(xr,yr)
# solver_FCI.construct_matrices()
# # solver_FCI.animate()

# solver_FCI.update_loop()
# # solver_FCI.show_recorder()
# _,H_num_FCI,_=solver_FCI.analytical_sol(p_all=False,f_lim=Wc+4/width)

# plt.plot(w,H_ana,color='orange',label='Analytical')
# plt.plot(w,H_num_Yee,'x',color='blue',label='Yee')
# plt.plot(w,H_num_FCI,'+',color='red',label='FCI')
# plt.xlabel(r'Angular frequency $\omega$ [$\frac{rad}{s}$]')
# plt.ylabel(r'Transfer function H($\omega$)')
# plt.grid()
# plt.legend()
# plt.show()

