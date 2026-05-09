import numpy as np
import matplotlib.pyplot as plt
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e

a = 15e-9
b = 5e-9
Lx = (3*a+2*b+40e-9) # Extra space for barrier to not have an influence
Ly = 40e-9
Lz = Ly
dx = Lx/3000
U0 = 0.6*e.value
CFL=0.99
sigma_x = (a/4)
x0 = 4*sigma_x
xr = (9*a/4+2*b+20e-9)

m = 1
n = 1

m_eff = 0.023*m_e.value
dt=CFL*2/(2*hbar.value/m_eff*(1/dx**2)+1/hbar.value*U0)
E = 6*e.value
print(f'Energy: {E/e.value} eV')
kx = np.sqrt(2*m_eff*E/hbar.value**2)
N_layer = 300
alpha = 1
sigma = alpha * hbar.value / (dt * N_layer)
k = 4 # exponent for the absorbing boundary strength
t_max = 3*Lx*np.sqrt(m_eff/2/E)

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

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,CFL=CFL,order=2,ABC=True)
# solver.add_recorder(xr)
# solver.animate(speed=1000)

# solver.update_loop()
# solver.show_recorder()

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

###
# Current density:
###

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
# Spectral content of the source:
###

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)
# solver.spectral_source(2)

###
# Validation with analytical solution:
###

solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=4,ABC=True)
Nt=solver.Nt
solver.add_barriers(U0)
solver.add_recorder(xr)

E_ana,T_ana = solver.analytical_T()

# solver.update_loop()
# solver.show_recorder()
# E_num, psi_bar = solver.psi_freq(np.pad(np.array(solver.psiRe_record_left),(0,5*Nt),'constant'),np.pad(np.array(solver.psiIm_record_left),(0,5*Nt),'constant'))
# psi_bar_sq = np.abs(psi_bar)**2

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=4,ABC=True)
# solver.add_recorder(xr)

# solver.update_loop()
# solver.show_recorder()
# _, psi_free = solver.psi_freq(np.pad(np.array(solver.psiRe_record_left),(0,5*Nt),'constant'),np.pad(np.array(solver.psiIm_record_left),(0,5*Nt),'constant'))
# psi_free_sq = np.abs(psi_free)**2
# mask=(E_num/e.value<((E+kx*hbar.value**2/sigma_x/m_eff)/e.value)) & (E_num/e.value>((E-kx*hbar.value**2/sigma_x/m_eff)/e.value))
# # mask=(E_num/e.value<7)&(E_num/e.value>5)

# T_num = psi_bar_sq[mask]/psi_free_sq[mask]
# plt.plot(E_num[mask]/e.value,T_num,label='Numerical')
plt.plot(np.real(E_ana)/e.value,T_ana,label='Analytical')
plt.xlabel('Energy [eV]')
plt.ylabel('Transmission')
# plt.xlim(0,0.6)
# plt.ylim(0,1)
plt.legend()
plt.show()

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