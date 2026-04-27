import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c,epsilon_0, mu_0
from Class_FCI import FCI
from Class_Yee import Yee
import time

# In this file, we will build a Fabry-Perot interferrometer and analyze the relative power reflection coefficient. We first start
# with a simple Fabry-Perot made out of glass. We will first look if the Fabry-Perot filter works by varying the thickness of this filter.
# If the frequency shifts match the analytical frequency shifts, we can confidently say that the filter works. We first do this for Yee.
# We start by defining our solver parameters and source parameters.

# L = 1
# Nx = 351
# Ny = 351
# CFL = 0.9
# dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
# J0 = 1000
# Wc = 0.35/dt
# width = 5/Wc
# tc = 5*width
# Nt = int(30*tc/dt)
# N_PML = 70  
# sigma_max=1.83279
# kappa_max=3.221

# We define our material. We take the refractive index of glass as n=1.33

# n = 1.33
# eps = n**2

# We put our recorder on the source, we will extract the initial created field by the source later to accuratly describe the relfection
# coefficient.

# xs = Nx//4
# ys = Ny//2
# xr = Nx//4
# yr = Ny//2

# fig,axes = plt.subplots(4,1,figsize=(6,10))
# i = 0

# We now let the simulation run without a material, so we can substract the initially generated field.

# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)
# solver.update_loop()
# Ez_applied=np.array(solver.recorded_Ez)

# We now calculate the reflection coefficient for various thicknesses

# colors = ['blue', 'orange', 'green', 'red']
# for d in [10,20,30,40]:
    
#     delta = c/(2*n*d*L/Nx)
#     print(r'Expected peak seperation for d = '+f"{d*L/Nx*10**3:.2f}"+r' mm [GHz]: '+f"{delta/10**9:.2f}")

#     solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
#     solver.add_source(xs,ys,J0,tc,width,Wc)
#     solver.add_recorder(xr,yr)
#     solver.add_material(Nx-N_PML-3*d//2-d//2,Nx-N_PML-3*d//2+d//2,0,Ny,eps_r=eps,mu_r=1,sigma=0)
#     # solver.animate(speed=10)
#     solver.update_loop()
#     # solver.show_recorder()

#     padded_Ez_shielded = np.pad(np.array(solver.recorded_Ez)-Ez_applied, (0, 5*Nt), 'constant')
#     Ez_shielded = np.fft.rfft(padded_Ez_shielded)
#     padded_source = np.pad(Ez_applied, (0, 5*Nt), 'constant')
#     spec_source = np.fft.rfft(padded_source)

#     f=np.fft.rfftfreq(len(padded_Ez_shielded),dt)
#     mask = np.where((2*np.pi*f<Wc+4/width) & (2*np.pi*f>Wc-4/width))
#     R=np.abs(Ez_shielded/spec_source)**2
#     axes[i].plot(f[mask]/10**9,R[mask],color=colors[i],label=f'd={d/Nx*L*1e3:.0f} mm')
#     axes[i].legend()
#     axes[i].set_ylabel(r'R [/]')
#     i+=1
    
# axes[0].set_title(r'Normalized Power Reflection Coefficient for a glass Fabry-Perot etalon')
# axes[3].set_xlabel(r'Frequency [GHz]')
# plt.tight_layout()
# plt.show()

# We now do this again for FCI, we use the same source parameters as Yee.

# Nx = 251
# Ny = 251
# CFL = 0.9
# dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
# J0 = 1000
# Wc = 0.25/dt
# width = 5/Wc
# tc = 5*width
# Nt = int(10*tc/dt)

# dx=L*np.ones(Nx)/Nx
# dy=L*np.ones(Ny)/Ny

# solver = FCI(Nt,dx,dy,dt,k_max=1,sigma_max=10,drude=False)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)
# solver.construct_matrices()
# solver.update_loop()
# Ez_applied=np.array(solver.recorded_Ez)

# fig,axes = plt.subplots(4,1,figsize=(6,10))
# i = 0

# colors = ['blue', 'orange', 'green', 'red']
# for d in [30,40,50,60]:
    
#     delta = c/(2*n*d*L/Nx)
#     print(r'Expected peak seperation for d = '+f"{d*L/Nx*10**3:.2f}"+r' mm [GHz]: '+f"{delta/10**9:.2f}")

#     dx=L/Nx*np.ones(Nx)
#     dy=L/Ny*np.ones(Ny)

#     solver = FCI(Nt,dx,dy,dt,k_max=1,sigma_max=10,drude=False)
#     solver.add_source(xs,ys,J0,tc,width,Wc)
#     solver.add_recorder(xr,yr)
#     solver.add_material(Nx-10-3*d//2-d//2,Nx-10-3*d//2+d//2,0,Ny,eps_r=eps,mu_r=1,sigma=0)
#     solver.construct_matrices()
#     solver.update_loop()
#     # solver.show_recorder()

#     padded_Ez_shielded = np.pad(np.array(solver.recorded_Ez)-Ez_applied, (0, 5*Nt), 'constant')
#     Ez_shielded = np.fft.rfft(padded_Ez_shielded)
#     padded_source = np.pad(Ez_applied, (0, 5*Nt), 'constant')
#     spec_source = np.fft.rfft(padded_source)

#     f=np.fft.rfftfreq(len(padded_Ez_shielded),dt)
#     mask = np.where((2*np.pi*f<Wc+4/width) & (2*np.pi*f>Wc-4/width))
#     R=np.abs(Ez_shielded/spec_source)**2
#     axes[i].plot(f[mask]/10**9,R[mask],color=colors[i],label=f'd={d/Nx*L*1e3:.0f} mm')
#     axes[i].legend()
#     axes[i].set_ylabel(r'R [/]')
#     i+=1
    
# axes[0].set_title(r'Normalized Power Reflection Coefficient for a glass Fabry-Perot etalon')
# axes[3].set_xlabel(r'Frequency [GHz]')
# plt.tight_layout()
# plt.show()  

# We see that both Yee and produce the correct frequency seprations, validating the set-up. We can now do some more exotic measurements.
# We can also study the effect of the Finesse on this medium, the results are shown here.

# L = 1
# Nx = 401
# Ny = 401
# CFL = 0.9
# dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
# J0 = 1000
# Wc = 0.35/dt
# width = 5/Wc
# tc = 5*width
# Nt = int(40*tc/dt)
# N_PML = 70  
# sigma_max=1.83279
# kappa_max=3.221

# xs=Nx//5
# ys=Ny//2
# xr=Nx//5
# yr=Ny//2

# n=1.33
# d=40

# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)
# solver.update_loop()
# # solver.show_recorder()
# Ez_applied=np.array(solver.recorded_Ez)

# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)
# solver.add_material(Nx-N_PML-3*d//2-d//2,Nx-N_PML-3*d//2+d//2,0,Ny,eps_r=n**2,mu_r=1,sigma=0)
# # solver.animate(speed=10)
# solver.update_loop()
# # solver.show_recorder()

# padded_Ez_shielded1 = np.pad(np.array(solver.recorded_Ez)-Ez_applied, (0, 5*Nt), 'constant')
# Ez_shielded1 = np.fft.rfft(padded_Ez_shielded1)
# padded_source1 = np.pad(Ez_applied, (0, 5*Nt), 'constant')
# spec_source1 = np.fft.rfft(padded_source1)

# f=np.fft.rfftfreq(len(padded_Ez_shielded1),dt)
# mask1 = np.where((2*np.pi*f<Wc+3/width) & (2*np.pi*f>Wc-3/width))
# R1=np.abs(Ez_shielded1/spec_source1)**2

# L = 1
# Nx = 401
# Ny = 401
# CFL = 0.9
# dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
# J0 = 1000
# Wc = 0.35/dt
# width = 5/Wc
# tc = 5*width
# Nt = int(40*tc/dt)
# N_PML = 70  
# sigma_max=1.83279
# kappa_max=3.221

# n=1.33*40/36
# d=36

# solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)
# solver.add_material(Nx-N_PML-3*d//2-d//2,Nx-N_PML-3*d//2+d//2,0,Ny,eps_r=n**2,mu_r=1,sigma=0)
# # solver.animate(speed=10)
# solver.update_loop()
# # solver.show_recorder()

# padded_Ez_shielded2 = np.pad(np.array(solver.recorded_Ez)-Ez_applied, (0, 5*Nt), 'constant')
# Ez_shielded2 = np.fft.rfft(padded_Ez_shielded2)
# padded_source2 = np.pad(Ez_applied, (0, 5*Nt), 'constant')
# spec_source2 = np.fft.rfft(padded_source2)

# mask2 = np.where((2*np.pi*f<Wc+3/width) & (2*np.pi*f>Wc-3/width))
# R2=np.abs(Ez_shielded2/spec_source2)**2

# plt.plot(f[mask1]/10**9,R1[mask1],color='green')
# plt.plot(f[mask2]/10**9,R2[mask2],color='red')
# plt.xlabel(r'Frequency f [GHz]')
# plt.ylabel(r'Normalized power reflection coefficient')
# plt.grid()
# plt.show()

# We now also compare Yee and FCI for generating the reflection coefficient of this Fabry-Perot filter. For the same thicnkess d=40
# for a glass plate n=1.33. We will do this by plotting the analytical frequencies where we expect maxima and comparing how well both
# Yee and FCI follow these peaks

L = 1
Nx = 301
Ny = 301
CFL = 0.9
dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)
J0 = 1000
Wc = 0.25/dt
width = 5/Wc
tc = 5*width
Nt = int(10*tc/dt)
N_PML = 60  
sigma_max=1.83279
kappa_max=3.221

xs=Nx//5
ys=Ny//2
xr=Nx//5
yr=Ny//2

solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)
solver.update_loop()
# solver.show_recorder()
Ez_applied=np.array(solver.recorded_Ez)

n=1.33
d=40

solver = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)
solver.add_material(Nx-N_PML-3*d//2-d//2,Nx-N_PML-3*d//2+d//2,0,Ny,eps_r=n**2,mu_r=1,sigma=0)
# solver.animate(speed=10)
solver.update_loop()
# solver.show_recorder()

padded_Ez_shielded = np.pad(np.array(solver.recorded_Ez)-Ez_applied, (0, 5*Nt), 'constant')
Ez_shielded = np.fft.rfft(padded_Ez_shielded)
padded_source = np.pad(Ez_applied, (0, 5*Nt), 'constant')
spec_source = np.fft.rfft(padded_source)

f=np.fft.rfftfreq(len(padded_Ez_shielded),dt)
mask = np.where((2*np.pi*f<Wc+4/width) & (2*np.pi*f>Wc-4/width))
R_Yee=np.abs(Ez_shielded/spec_source)**2

dx=np.ones(Nx)*L/Nx
dy=np.ones(Ny)*L/Ny

solver = FCI(Nt,dx,dy,dt,k_max=2,sigma_max=10,drude=False)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)
solver.construct_matrices()
# solver.animate(speed=2)
solver.update_loop()
# solver.show_recorder()
Ez_applied=np.array(solver.recorded_Ez)

solver = FCI(Nt,dx,dy,dt,k_max=1,sigma_max=10,drude=False)
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)
solver.add_material(Nx-N_PML-3*d//2-d//2,Nx-N_PML-3*d//2+d//2,0,Ny,eps_r=n**2,mu_r=1,sigma=0)
solver.construct_matrices()
solver.update_loop()
# solver.show_recorder()

padded_Ez_shielded = np.pad(np.array(solver.recorded_Ez)-Ez_applied, (0, 5*Nt), 'constant')
Ez_shielded = np.fft.rfft(padded_Ez_shielded)
padded_source = np.pad(Ez_applied, (0, 5*Nt), 'constant')
spec_source = np.fft.rfft(padded_source)

f=np.fft.rfftfreq(len(padded_Ez_shielded),dt)
mask = np.where((2*np.pi*f<Wc+4/width) & (2*np.pi*f>Wc-4/width))
R_FCI=np.abs(Ez_shielded/spec_source)**2

plt.plot(f[mask]/10**9,R_Yee[mask],color='blue',label='Yee')
plt.plot(f[mask]/10**9,R_FCI[mask],color='red',label='FCI')
plt.xlabel(r'Frequency f [GHz]')
plt.ylabel(r'Normalized power Reflection Coefficient R [/]')
plt.legend()
plt.grid()
plt.show()