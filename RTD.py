import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e
import cmath

a = 15
b = 5
Lx = 3*a+2*b+10 # Extra space for barrier to not have an influence
Ly = 40
Lz = Ly
dx = Lx/3000
U0 = 0.6*e.value*10**(-18)
dt = 0.7*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
x0 = a/3
sigma_x = a/10
xr = 7*a/3+2*b

m = 1
n = 1

m_eff = 0.023*m_e.value
E = hbar.value**2/(2*m_eff)*((np.pi*n/Ly)**2+(np.pi*m/Lz)**2)
print(f'Energy: {E/e.value*10**18:.2f} eV')
kx = np.sqrt(2*m_eff*E/hbar.value**2) # in 1/nm
sigma = 5*E
k = 3 # exponent for the absorbing boundary strength
N_layer = 200
t_max = 20000/kx
speed = int(t_max*Lx/500)

###
# Without Absorbing Boundaries:
###

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=False)
# solver.add_recorder(xr)
# solver.animate(speed = speed)

###
# With Absorbing Boundaries:
###

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True)
# solver.add_recorder(xr)
# solver.animate(speed = speed)

###
# With potential barriers:
###

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True)

# solver.add_barriers(U0)
# solver.add_recorder(xr)
# solver.animate(speed = speed)

# solver.restart()
# solver.update_loop_2()
# solver.show_recorder()

# # Current density:

# t, J_time = solver.J_time()
# plt.plot(t,J_time)
# plt.xlabel('Time [s]')
# plt.ylabel('Current density')
# plt.show()

# E, J_freq = solver.J_freq(t,J_time)
# plt.plot(E,np.abs(J_freq))
# plt.xlabel('Energy [eV]')
# plt.ylabel('Current density')
# plt.show()

###
# Validation with analytical solution:
###

solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True)
solver.add_barriers(U0)
solver.add_recorder(xr)

E,T_ana = solver.analytical_T()
plt.plot(np.real(E)/e.value*10**18,T_ana,label='Analytical')
plt.xlabel('Energy [eV]')
plt.ylabel('Transmission')
plt.legend()
plt.show()