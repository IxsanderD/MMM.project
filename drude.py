import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee
from Class_FCI import FCI
from scipy.constants import c

# In this file, the effect of a Drude material will be compared with a normal material for both Yee and FCI

# YEE

### Coding parameters
L = 1
Nx = 201
Ny = 201
c = c
CFL = 0.9
dt = CFL/c/np.sqrt(1/(L/Nx)**2+1/(L/Ny)**2)

### Source parameters
J0 = 50
Wc = 0.35/dt
width = 5/Wc
tc = 5*width
tf = 12*tc
Nt = int(tf/dt)
print('Central angular frequency: ', Wc*1e-9, ' GHz')

### Source position
xs = Nx//4
ys = Ny//2

### Recorder position
xr = 3*Nx//4
yr = Ny//2

### PML parameters
N_PML = 30
sigma_max=1.83279
kappa_max=3.221

### Material parameters
R = 0.1 # Width of the slab
x_start = int(Nx//2 - (R*Nx/L)//2)
x_einde = int(Nx//2 + (R*Nx/L)//2)
y_start = int(N_PML//4)
y_einde = int(Ny - N_PML//4)
eps_r = 1
sigma_DC = 0.1
gamma = 1/Wc
print('Gamma: ',gamma)

### First we add a normal conducting material
### Run Yee
solver_Yee = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
solver_Yee.add_recorder(xr,yr)
solver_Yee.add_drude_material(x_start,x_einde,y_start,y_einde,eps_r,sigma_DC,0)
solver_Yee.update_loop()

### Visualise
# solver_Yee.show_recorder()
# # solver_Yee.restart()
# solver_Yee.animate()

### Extract results
Ez_Yee_normal = solver_Yee.recorded_Ez
time_Yee_normal = np.arange(solver_Yee.Nt*6)*solver_Yee.dt
Ez_Yee_normal = np.pad(Ez_Yee_normal,(0,5*solver_Yee.Nt),'constant')
Ez_freq_Yee_normal = np.fft.rfft(Ez_Yee_normal)
freq_Yee_normal = np.fft.rfftfreq(6*solver_Yee.Nt,solver_Yee.dt)

### Now we use gamma so it behaves like a drude material

### Run Yee
solver_Yee = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
solver_Yee.add_recorder(xr,yr)
solver_Yee.add_drude_material(x_start,x_einde,y_start,y_einde,eps_r,sigma_DC,gamma)
solver_Yee.update_loop()

### Visualise
# solver_Yee.show_recorder()
# # solver_Yee.restart()
# solver_Yee.animate()

### Extract results
Ez_Yee_drude = solver_Yee.recorded_Ez
time_Yee_drude = np.arange(6*solver_Yee.Nt)*solver_Yee.dt
Ez_Yee_drude = np.pad(Ez_Yee_drude,(0,5*solver_Yee.Nt),'constant')
Ez_freq_Yee_drude = np.fft.rfft(Ez_Yee_drude)
freq_Yee_drude = np.fft.rfftfreq(6*solver_Yee.Nt,solver_Yee.dt)

# FCI

### Coding parameters
dx=np.ones(Nx)*L/Nx
dy=np.ones(Ny)*L/Ny

### Grid refinement
factor = 5
x_einde = int(x_start + R*Nx/L*factor)
xr = xr + int(R*Nx/L*(factor-1))
Nx = 2*x_start + int(R*Nx/L*factor)
dx = np.ones(Nx)*(L+(factor-1)*R)/Nx
dx[x_start:x_einde] = dx[x_start:x_einde]/factor

### PML parameters
k_max=1
sigma_max=1

### First we investigate again the normal material
### Run FCI
solver_FCI=FCI(Nt,dx,dy,dt,k_max,sigma_max,drude=True)
solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
solver_FCI.add_recorder(xr,yr)
solver_FCI.add_material(x_start,x_einde,y_start,y_einde,eps_r,1,sigma_DC,0)
solver_FCI.construct_matrices()
solver_FCI.update_loop_drude()

### Visualise
# solver_FCI.show_recorder()
# # solver_FCI.restart()
# solver_FCI.animate()

### Extract results
Ez_FCI_normal = solver_FCI.recorded_Ez
time_FCI_normal = np.arange(6*solver_FCI.Nt)*solver_FCI.dt
Ez_FCI_normal = np.pad(Ez_FCI_normal,(0,5*solver_FCI.Nt),'constant')
Ez_freq_FCI_normal = np.fft.rfft(Ez_FCI_normal)
freq_FCI_normal = np.fft.rfftfreq(6*solver_FCI.Nt,solver_FCI.dt)

### Lastly we again investigate the drude material
### Run FCI
solver_FCI=FCI(Nt,dx,dy,dt,k_max,sigma_max,drude=True)
solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
solver_FCI.add_recorder(xr,yr)
solver_FCI.add_material(x_start,x_einde,y_start,y_einde,eps_r,1,sigma_DC,gamma)
solver_FCI.construct_matrices()
solver_FCI.update_loop_drude()

### Visualise
# solver_FCI.show_recorder()
# # solver_FCI.restart()
# solver_FCI.animate()

### Extract results
Ez_FCI_drude = solver_FCI.recorded_Ez
time_FCI_drude = np.arange(6*solver_FCI.Nt)*solver_FCI.dt
Ez_FCI_drude = np.pad(Ez_FCI_drude,(0,5*solver_FCI.Nt),'constant')
Ez_freq_FCI_drude = np.fft.rfft(Ez_FCI_drude)
freq_FCI_drude = np.fft.rfftfreq(6*solver_FCI.Nt,solver_FCI.dt)

### Plot extracted results

### Plot the time domain results
plt.plot(time_Yee_normal,Ez_Yee_normal,label = 'Normal conductor (Yee)', linestyle = '--')
plt.plot(time_Yee_drude,Ez_Yee_drude,label = 'Drude material (Yee)', linestyle = '-')
plt.plot(time_FCI_normal,Ez_FCI_normal,label = 'Normal conductor (FCI)', linestyle = '--')
plt.plot(time_FCI_drude,Ez_FCI_drude,label = 'Drude material (FCI)', linestyle = '-')
plt.xlabel('Time (s)')
plt.ylabel('Electric field (V/m)')
plt.title('Recorded electric field behind a conductor and drude material slab')
plt.legend()
plt.show()

band = (Wc + 5/width)/2/np.pi
mask = freq_Yee_normal < band

### Plot the transfer function
plt.plot(freq_Yee_drude[mask]*1e-9,np.abs(Ez_freq_Yee_drude/Ez_freq_Yee_normal)[mask],label = 'Drude material (Yee)')
plt.plot(freq_FCI_drude[mask]*1e-9,np.abs(Ez_freq_FCI_drude/Ez_freq_FCI_normal)[mask],label = 'Drude material (FCI)')
plt.plot(Wc/2/np.pi*1e-9**np.ones(2),np.linspace(np.min(np.abs(Ez_freq_Yee_drude/Ez_freq_Yee_normal)[mask]),np.max(np.abs(Ez_freq_Yee_drude/Ez_freq_Yee_normal)[mask]),2),label = 'Cut-off frequency $f_c$',linewidth=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Relative transmission $T_r$')
plt.grid(True,which='both')
plt.legend()
plt.show()