import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from Class_FCI import FCI
from Class_Yee import Yee

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

# xs = Nx//4
# ys = 3*Ny//4
# xr = Nx//4
# yr = Ny//4

# N_PML = 40  

# sigma_c=5.96*10**7

# fig,axes = plt.subplots(4,1,figsize=(6,10))
# i = 0
# colors = ['blue', 'orange', 'green', 'red']
# for d in [30,40,50,60]:
    
#     xs = Nx//4
#     ys = Ny//2
#     xr = Nx//4+d//2+2
#     yr = Ny//2

#     solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True)
#     solver.add_source(xs,ys,J0,tc,width,Wc)
#     solver.add_recorder(xr,yr)
#     # solver.animate(speed = 10)
#     solver.update_loop()
#     # solver.show_recorder()

#     Ez_unshielded = np.fft.rfft(solver.recorded_Ez)

#     solver.restart()
#     solver.add_source(xs,ys,J0,tc,width,Wc)
#     solver.add_recorder(xr,yr)
#     solver.add_material(Nx//2-d//2,Nx//2+d//2,10,Ny-10,eps_r=1,mu_r=1,sigma=sigma_c)
#     # solver.animate(speed = 10)
#     solver.update_loop()
#     # solver.show_recorder()

#     Ez_shielded = np.fft.rfft(solver.recorded_Ez)

#     f=np.fft.rfftfreq(Nt,dt)
#     SE=20*np.log10(np.abs(Ez_unshielded/Ez_shielded))
#     axes[i].plot(2*np.pi*f,SE,color=colors[i],label=f'd={d} mm')
#     axes[i].legend()
#     axes[i].set_ylabel(r'SE [dB]')
#     i+=1
    
# axes[0].set_title(r'SE of copper')
# axes[3].set_xlabel(r'Angular frequency $\omega$')
# axes[0].set_xticklabels([])
# axes[1].set_xticklabels([])
# axes[2].set_xticklabels([])
# plt.tight_layout()
# plt.show()

Lx = 1
Nx = 151
Ny = 41
Nt = 400
c = 1
J0 = 10
width = 5/Wc
tc = 5*width
dt = 10*tc/Nt

sigma_c=5.96*10**7

xs = Nx//4
ys = Ny//2
xr = Nx//4
yr = Ny//2

fig,axes = plt.subplots(4,1,figsize=(6,10))
i = 0
colors = ['blue', 'orange', 'green', 'red']

for d in [30]:
    
    xs = Nx//4
    ys = Ny//2
    xr = Nx//4+d//2+2
    yr = Ny//2

    dx=np.ones(Nx)*(Lx-d/300)/Nx
    dx[Nx//2-d//2:Nx//2+d//2]=1/300
    dy=np.ones(Ny)*(Lx-d/300)/Nx

    solver = FCI(Nt,dx,dy,dt,1,1000)
    solver.add_source(xs,ys,J0,tc,width,Wc)
    solver.add_recorder(xr,yr)
    solver.construct_matrices()
    # solver.animate(speed = 10)
    solver.update_loop()
    # solver.show_recorder()

    Ez_unshielded = np.fft.rfft(solver.recorded_Ez)

    solver.restart()
    solver.add_source(xs,ys,J0,tc,width,Wc)
    solver.add_recorder(xr,yr)
    solver.add_material(Nx//2-d//2,Nx//2+d//2,10,Ny-10,eps_r=1,mu_r=1,sigma=sigma_c)
    solver.construct_matrices()
    # solver.animate(speed = 10)
    solver.update_loop()
    # solver.show_recorder()

    Ez_shielded = np.fft.rfft(solver.recorded_Ez)

    f=np.fft.rfftfreq(Nt,dt)
    SE=20*np.log10(np.abs(Ez_unshielded/Ez_shielded))
    axes[i].plot(2*np.pi*f,SE,color=colors[i],label=f'd={d} mm')
    axes[i].legend()
    axes[i].set_ylabel(r'SE [dB]')
    i+=1
    
axes[0].set_title(r'SE of copper')
axes[3].set_xlabel(r'Angular frequency $\omega$')
axes[0].set_xticklabels([])
axes[1].set_xticklabels([])
axes[2].set_xticklabels([])
plt.tight_layout()
plt.show()
