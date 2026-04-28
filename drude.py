import numpy as np
import matplotlib.pyplot as plt
from Class_Yee import Yee
from Class_FCI import FCI
from scipy.constants import c

# In this file, the effect of a Drude material will be compared with a normal material for both codes

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
Wc = 0.15/dt
width = 3/Wc
tc = 5*width
tf = 5*tc
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
R = 0.1
x_start = int(Nx//2 - (R*Nx/L)//2)
x_einde = int(Nx//2 + (R*Nx/L)//2)
y_start = int(N_PML//4)
y_einde = int(Ny - N_PML//4)
eps_r = 1
sigma_DC = 111
gamma = 0

# ### Run Yee
# solver_Yee = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
# solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
# solver_Yee.add_recorder(xr,yr)
# solver_Yee.add_drude_material(x_start,x_einde,y_start,y_einde,eps_r,sigma_DC,gamma)
# solver_Yee.update_loop()

# ### Visualise
# # solver_Yee.show_recorder()
# # solver_Yee.animate()

# ### Extract results
# Ez_Yee_normal = solver_Yee.recorded_Ez
# time_Yee_normal = np.arange(solver_Yee.Nt)*solver_Yee.dt
# Ez_freq_Yee_normal = np.fft.rfft(Ez_Yee_normal)
# freq_Yee_normal = np.fft.rfftfreq(solver_Yee.Nt,solver_Yee.dt)

# ### Now we change gamma so it becomes a drude material
# gamma = 4.5

# ### Run Yee
# solver_Yee = Yee(L,Nx,Ny,Nt,dt,N_PML,PML=True,sigma_max=sigma_max,kappa_max=kappa_max)
# solver_Yee.add_source(xs,ys,J0,tc,width,Wc)
# solver_Yee.add_recorder(xr,yr)
# solver_Yee.add_drude_material(x_start,x_einde,y_start,y_einde,eps_r,sigma_DC,gamma)
# solver_Yee.update_loop()

# ### Visualise
# # solver_Yee.show_recorder()
# # solver_Yee.animate()

# ### Extract results
# Ez_Yee_drude = solver_Yee.recorded_Ez
# time_Yee_drude = np.arange(solver_Yee.Nt)*solver_Yee.dt
# Ez_freq_Yee_drude = np.fft.rfft(Ez_Yee_drude)
# freq_Yee_drude = np.fft.rfftfreq(solver_Yee.Nt,solver_Yee.dt)

# FCI

### Coding parameters
dx=np.ones(Nx)*L/Nx
dy=np.ones(Ny)*L/Ny
# Nt = 100
# dt = tf/Nt

### Grid refinement
# factor = 5
# x_einde = int(x_start + R*Nx/L*factor)
# xr = xr + int(R*Nx/L*(factor-1))
# Nx = 2*x_start + int(R*Nx/L*factor)
# dx = np.ones(Nx)*(L+(factor-1)*R)/Nx
# dx[x_start:x_einde] = dx[x_start:x_einde]/factor

### PML parameters
k_max=1
sigma_max=10

### First we investigate again the normal material
gamma = 0

### Run FCI
solver_FCI=FCI(Nt,dx,dy,dt,k_max,sigma_max,drude=True)
solver_FCI.construct_matrices()
solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
solver_FCI.add_recorder(xr,yr)
# solver_FCI.add_material(x_start,x_einde,y_start,y_einde,eps_r,1,sigma_DC,gamma)
solver_FCI.update_loop_drude()

### Visualise
solver_FCI.show_recorder()
solver_FCI.animate()

### Extract results
Ez_FCI_normal = solver_FCI.recorded_Ez
time_FCI_normal = np.arange(solver_FCI.Nt)*solver_FCI.dt
Ez_freq_FCI_normal = np.fft.rfft(Ez_FCI_normal)
freq_FCI_normal = np.fft.rfftfreq(solver_FCI.Nt,solver_FCI.dt)

### Now we again change gamma so it becomes a drude material
gamma = 4.5

### Run FCI
# solver_FCI=FCI(Nt,dx,dy,dt,k_max,sigma_max,drude=True)
# solver_FCI.construct_matrices()
# solver_FCI.add_source(xs,ys,J0,tc,width,Wc)
# solver_FCI.add_recorder(xr,yr)
# solver_FCI.add_material(x_start,x_einde,y_start,y_einde,eps_r,1,sigma_DC,gamma)
# solver_FCI.update_loop_drude()

# ### Visualise
# solver_FCI.show_recorder()
# solver_FCI.animate()

# ### Extract results
# Ez_FCI_drude = solver_FCI.recorded_Ez
# time_FCI_drude = np.arange(solver_FCI.Nt)*solver_FCI.dt
# Ez_freq_FCI_drude = np.fft.rfft(Ez_FCI_drude)
# freq_FCI_drude = np.fft.rfftfreq(solver_FCI.Nt,solver_FCI.dt)

# ### Plot the time domain results
# plt.plot(time_Yee_normal,Ez_Yee_normal,label = 'Normal conductor (Yee)', linestyle = '--')
# plt.plot(time_Yee_drude,Ez_Yee_drude,label = 'Drude material (Yee)', linestyle = '-')
# plt.plot(time_FCI_normal,Ez_FCI_normal,label = 'Normal conductor (FCI)', linestyle = '--')
# plt.plot(time_FCI_drude,Ez_FCI_drude,label = 'Drude material (FCI)', linestyle = '-')
# plt.xlabel('Time (s)')
# plt.ylabel('Electric field (V/m)')
# plt.title('Recorded electric field behind a conductor and drude material slab')
# plt.legend()
# plt.show()

# ### Plot the frequency domain results
# plt.plot(freq_Yee_normal,Ez_freq_Yee_normal, label = 'Normal conductor (Yee)', linestyle = '--')
# plt.plot(freq_Yee_drude,Ez_freq_Yee_drude,label = 'Drude material (Yee)', linestyle = '-')
# plt.plot(freq_FCI_normal,Ez_freq_FCI_normal,label = 'Normal conductor (FCI)', linestyle = '--')
# plt.plot(freq_FCI_drude,Ez_freq_FCI_drude,label = 'Drude material (FCI)', linestyle = '-')
# plt.xscale('log')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Electric field (V/m)')
# plt.title('Recorded electric field behind a conductor and drude material slab')
# plt.legend()
# plt.show()