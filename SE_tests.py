import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,epsilon_0, mu_0
from Class_FCI import FCI
from Class_Yee import Yee
import time

L = 0.3
Nx = 300
Ny = 300
CFL = 0.9
dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
J0 = 1000
Wc = 0.35/dt
width = 5/Wc
tc = 5*width
Nt = int(20*tc/dt)

N_PML = 40  

n = 1.5
eps = n**2

print(Wc/2/np.pi/10**9)

xs = Nx//4
ys = Ny//2
xr = 3*Nx//4
yr = Ny//2

sigma_max=1.83279
kappa_max=3.221
fig,axes = plt.subplots(4,1,figsize=(6,10))
i = 0
colors = ['blue', 'orange', 'green', 'red']
for d in [30,40,50,60]:
    
    delta = c/(2*n*d*L/Nx)
    print(delta/10**9)

    solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
    solver.add_source(xs,ys,J0,tc,width,Wc)
    solver.add_recorder(xr,yr)
    # solver.animate(speed = 10)
    solver.update_loop()
    # solver.show_recorder()

    padded_Ez_unshielded = np.pad(solver.recorded_Ez, (0, 5*Nt), 'constant')
    Ez_unshielded = np.fft.rfft(padded_Ez_unshielded)

    solver.restart()
    solver.add_source(xs,ys,J0,tc,width,Wc)
    solver.add_recorder(xr,yr)
    solver.add_material(Nx//2-d//2,Nx//2+d//2,0,Ny,eps_r=eps,mu_r=1,sigma=0)
    # solver.animate(speed = 10)
    solver.update_loop()
    # solver.show_recorder()

    padded_Ez_shielded = np.pad(solver.recorded_Ez, (0, 5*Nt), 'constant')
    Ez_shielded = np.fft.rfft(padded_Ez_shielded)

    f=np.fft.rfftfreq(len(padded_Ez_shielded),dt)
    mask = np.where((2*np.pi*f<Wc+4/width) & (2*np.pi*f>Wc-4/width))
    SE=20*np.log10(np.abs(Ez_unshielded/Ez_shielded))
    axes[i].plot(f[mask]/10**9,SE[mask],color=colors[i],label=f'd={d/Nx*L*1e3:.0f} mm')
    axes[i].legend()
    axes[i].set_ylabel(r'SE [dB]')
    i+=1
    
axes[0].set_title(r'SE')
axes[3].set_xlabel(r'Frequency [GHz]')
plt.tight_layout()
plt.show()

Lx = 0.3
Nx = 301
Ny = 301
Nt = 300
J0 = 1000
width = 5/Wc
tc = 5*width
tf= 12*tc
dt= tf/Nt
dx = np.ones(Nx)*Lx/Nx
dy = np.ones(Ny)*Lx/Ny

print(Wc/2/np.pi/10**9)

xs = Nx//4
ys = Ny//2
xr = 3*Nx//4
yr = Ny//2

fig,axes = plt.subplots(4,1,figsize=(6,10))
i = 0
colors = ['blue', 'orange', 'green', 'red']

for d in [30,40,50,60]:
    
    delta = c/(2*n*d*Lx/Nx)
    print(delta/10**9)

    solver = FCI(Nt,dx,dy,dt,1,1000)
    solver.add_source(xs,ys,J0,tc,width,Wc)
    solver.add_recorder(xr,yr)
    start=time.perf_counter()
    solver.construct_matrices()
    end=time.perf_counter()
    # solver.animate(speed = 10)
    solver.update_loop()
    # solver.show_recorder()

    padded_Ez_unshielded = np.pad(solver.recorded_Ez, (0, 5*Nt), 'constant')
    Ez_unshielded = np.fft.rfft(padded_Ez_unshielded)

    solver.restart()
    solver.add_source(xs,ys,J0,tc,width,Wc)
    solver.add_recorder(xr,yr)
    solver.add_material(Nx//2-d//2,Nx//2+d//2,0,Ny,eps_r=eps,mu_r=1,sigma=0)
    solver.construct_matrices()
    # solver.animate(speed = 10)
    solver.update_loop()
    # solver.show_recorder()

    padded_Ez_shielded = np.pad(solver.recorded_Ez, (0, 5*Nt), 'constant')
    Ez_shielded = np.fft.rfft(padded_Ez_shielded)

    f=np.fft.rfftfreq(len(padded_Ez_shielded),dt)
    mask = np.where((2*np.pi*f<Wc+4/width) & (2*np.pi*f>Wc-4/width))
    SE=20*np.log10(np.abs(Ez_unshielded/Ez_shielded))
    axes[i].plot(f[mask]/10**9,SE[mask],color=colors[i],label=f'd={d} mm')
    axes[i].legend()
    axes[i].set_ylabel(r'SE [dB]')
    i+=1
    
axes[0].set_title(r'SE')
axes[3].set_xlabel(r'Frequency [GHz]')
plt.tight_layout()
plt.show()