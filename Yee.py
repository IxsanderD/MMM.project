import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee
from scipy.special import hankel2

L = 2
Nx = 300
Ny = 300
c = 1
CFL = 0.9
dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
J0 = 10000
Wc = 0.35/dt
width = 5/Wc
tc = 5*width
Nt = int(20*tc/dt)

xs = Nx//4
ys = 3*Ny//4
xr = Nx//4
yr = Ny//4

N_PML = 20

###
# Without PML
###
# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=False)
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
# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)

# solver.show_PML() # PML profiles

# solver.animate(speed = 10)
# solver.restart()

# solver.update_loop()
# solver.show_recorder()
# solver.restart()

###
# Analytical verification
###

# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)
# solver.update_loop()

# solver.analytical_solution(frequency_limit=None)

###
# With matreial
###

solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True)
solver.add_source(xs,ys,J0,tc,width,Wc)

# # Plot source in time and frequency domain
# plt.plot(np.arange(solver.Nt)*solver.dt, solver.applied_source, label='Applied source')
# plt.xlabel('Time (s)')
# plt.ylabel('Applied source (A/m^2)')
# plt.title('Time domain applied source')
# plt.legend()
# plt.show()
# plt.plot(2*np.pi*np.fft.rfftfreq(solver.Nt, solver.dt), np.abs(np.fft.rfft(solver.applied_source)*dt), label='Applied source (frequency domain)')
# plt.xlabel('Frequency (rad/s)')
# plt.ylabel('Applied source (A/m^2)')
# plt.title('Frequency domain applied source')
# plt.legend()
# plt.show()
# solver.animate()

solver.add_recorder(xr,yr)
# solver.add_material(3*Nx//4,4*Nx//5,Ny//4,3*Ny//4,eps_r=1,mu_r=1,sigma=100)
solver.add_drude_material(3*Nx//4,4*Nx//5,Ny//4,3*Ny//4,eps_r=1,sigma_DC=100,gamma=0)

solver.animate(speed = 10)
# solver.restart()

solver.update_loop()
solver.show_recorder()
solver.restart()