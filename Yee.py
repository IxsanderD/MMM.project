import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee
from scipy.special import hankel2
from scipy.constants import c

L = 1
Nx = 200
Ny = 200
c = c
CFL = 0.9
dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
J0 = 10
Wc = 0.5/dt
width = 9*dt
tc = 5*width
Nt = int(20*tc/dt)

xs = Nx//4
ys = 3*Ny//4
xr = Nx//4
yr = Ny//4

N_PML = 50

print(f'f = {Wc/2/np.pi:.2f}')

###
# Without PML
###
# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=False)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)

# solver.animate(speed = 10)
# solver.restart()

# solver.update_loop()
# solver.show_recorder()
# solver.restart()

###
# With PML
###
solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)

# solver.show_PML() # PML profiles

solver.animate(speed = 10)
solver.restart()

solver.update_loop()
solver.show_recorder()
solver.restart()

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

# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True)
# solver.add_source(xs,ys,J0,tc,width,Wc)

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

# solver.add_recorder(xr,yr)
# solver.add_material(3*Nx//4,4*Nx//5,Ny//4,3*Ny//4,eps_r=1,mu_r=1,sigma=100)
# solver.add_drude_material(3*Nx//4,4*Nx//5,Ny//4,3*Ny//4,eps_r=1,sigma_DC=100,gamma=0)

# solver.animate(speed = 10)
# solver.restart()

# solver.update_loop()
# solver.show_recorder()
# solver.restart()

###
# Transmission
###

# n = 2
# eps = n**2

# fig,axes = plt.subplots(4,1,figsize=(6,10))
# i = 0
# colors = ['blue', 'orange', 'green', 'red']
# for d in [70,80,90,100]:
    
#     delta = c/(2*n*d*L/Nx)
#     print(delta)
    
#     xs = Nx//4
#     ys = Ny//2
#     xr = 3*Nx//4
#     yr = Ny//2

#     solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True)
#     solver.add_source(xs,ys,J0,tc,width,Wc)
#     solver.add_recorder(xr,yr)
#     # solver.animate(speed = 10)
#     solver.update_loop()
#     # solver.show_recorder()

#     padded_Ez_unshielded = np.pad(solver.recorded_Ez, (0, 5*Nt), 'constant')
#     Ez_unshielded = np.fft.rfft(padded_Ez_unshielded)

#     solver.restart()
#     solver.add_source(xs,ys,J0,tc,width,Wc)
#     solver.add_recorder(xr,yr)
#     solver.add_material(Nx//2-d//2,Nx//2+d//2,0,Ny,eps_r=eps,mu_r=1,sigma=0)
#     # solver.animate(speed = 10)
#     solver.update_loop()
#     # solver.show_recorder()

#     padded_Ez_shielded = np.pad(solver.recorded_Ez, (0, 5*Nt), 'constant')
#     Ez_shielded = np.fft.rfft(padded_Ez_shielded)

#     f=np.fft.rfftfreq(len(padded_Ez_unshielded),dt)
#     # SE=20*np.log10(np.abs(Ez_unshielded/Ez_shielded))
#     T = np.abs(Ez_shielded/Ez_unshielded)**2
#     axes[i].plot(f,T,color=colors[i],label=f'd={d/Nx*L*1e3:.0f} mm')
#     axes[i].legend()
#     axes[i].set_ylabel(r'Transmission')
#     axes[i].set_xlim(Wc/2/np.pi-3/2/np.pi/width,Wc/2/np.pi+3/2/np.pi/width)
#     axes[i].set_ylim(0,5)
#     i+=1
    
# axes[0].set_title(r'Transmission')
# axes[3].set_xlabel(r'Frequency f [Hz]')
# plt.tight_layout()
# plt.show()