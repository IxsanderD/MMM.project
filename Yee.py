import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee
from scipy.special import hankel2

L = 1
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
# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,m,PML=True)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)

# # solver.show_PML() # PML profiles

# solver.animate(speed = 10)
# solver.restart()

# solver.update_loop()
# solver.show_recorder()
# solver.restart()

###
# Analytical verification
###

solver = Yee(L,Nx,Ny,Nt,dt,N_PML,m,PML=True)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)
solver.update_loop()

E_freq_sim = np.fft.fft(solver.recorded_Ez)
freq = np.fft.fftfreq(len(solver.recorded_Ez), dt)
E_freq_source = np.fft.fft(solver.source_Ez)

E_freq_sim = E_freq_sim[:len(E_freq_sim)//2]
E_freq_source = E_freq_source[:len(E_freq_source)//2]
omega = 2*np.pi*freq[:len(freq)//2]

E_freq_ana = -J0*omega/4*hankel2(0, omega/c*np.sqrt((xr-xs)**2+(yr-ys)**2))

plt.plot(omega, np.abs(E_freq_sim), label='Simulated recorder')
plt.plot(omega, np.abs(E_freq_source), label='Source')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('|E_z|')
plt.legend()
plt.show()

plt.plot(omega, np.abs(E_freq_sim/E_freq_source), label='Numerical response')
# plt.plot(omega, np.abs(E_freq_ana), label='Analytical response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('|E_z|')
plt.legend()
plt.show()

###
# With matreial
###

# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,m,PML=True)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)
# solver.add_material(3*Nx//4,4*Nx//5,Ny//4,3*Ny//4,eps_r=1,mu_r=1,sigma=40000000)

# solver.animate(speed = 10)
# solver.restart()

# solver.update_loop()
# solver.show_recorder()
# solver.restart()