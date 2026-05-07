import numpy as np
import matplotlib.pyplot as plt
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e

a = 15e-9
b = 5e-9
Lx = (3*a+2*b+20e-9) # Extra space for barrier to not have an influence
Ly = 40e-9
Lz = Ly
dx = Lx/3000
U0 = 0.6*e.value
x0 = (a/2+5e-9)
sigma_x = (a/5)
xr = (7*a/3+2*b+10e-9)

m = 1
n = 1

m_eff = 0.023*m_e.value
E = 8*e.value
print(f'Energy: {E/e.value} eV')
kx = np.sqrt(2*m_eff*E/hbar.value**2)
N_layer = 200
sigma = np.log(100000)/N_layer/dx*E/kx
k = 4 # exponent for the absorbing boundary strength
t_max = 1.5*Lx*np.sqrt(m_eff/2/E)

print(f't_max: {t_max}')

###
# Without Absorbing Boundaries:
###

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=False)
# solver.add_recorder(xr)
# solver.animate(speed = 1000)

###
# With Absorbing Boundaries:
###

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)
# solver.add_recorder(xr)
# solver.animate(speed=1000)

###
# With potential barriers:
###

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)

# solver.add_barriers(U0)
# solver.add_recorder(xr)
# solver.animate(speed = 1000)
# solver.restart()

# solver.update_loop()
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

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)
# solver.add_barriers(U0)
# solver.add_recorder(xr)

# E_ana,T_ana = solver.analytical_T()

# solver.update_loop()
# solver.show_recorder()
# t, J_time = solver.J_time()
# E, J_barrier = solver.J_freq(t,J_time)

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)
# solver.add_recorder(xr)

# solver.update_loop()
# solver.show_recorder()
# t, J_time = solver.J_time()
# E_num, J_free = solver.J_freq(t,J_time)

# T_num = np.abs(J_barrier/J_free)
# plt.plot(E_num,T_num,label='Numerical')
# plt.plot(np.real(E_ana)/e.value,T_ana,label='Analytical')
# plt.xlabel('Energy [eV]')
# plt.ylabel('Transmission')
# # plt.xlim(0,0.6)
# # plt.ylim(0,1)
# plt.legend()
# plt.show()

###
# With potential V0
###

# V0 = 0.05*e.value

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)

# solver.add_barriers(U0)
# solver.add_potential(V0)
# solver.plot_potential()
# solver.add_recorder(xr)
# solver.animate(speed = 1000)
# solver.restart()

# Comparison of orders:

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)
# solver.add_barriers(U0)
# solver.add_recorder(xr)
# solver.update_loop()
# t,J = solver.J_time()
# plt.plot(t,J,label='Order 2')
# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=4,ABC=True)
# solver.add_barriers(U0)
# solver.add_recorder(xr)
# solver.update_loop()
# t,J = solver.J_time()
# plt.plot(t,J,label='Order 4')
# plt.legend()
# plt.show()