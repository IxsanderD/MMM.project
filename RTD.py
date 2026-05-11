import numpy as np
import matplotlib.pyplot as plt
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e

a = 15e-9
b = 5e-9
Lx = (3*a+2*b+108e-9) # Extra space for barrier to not have an influence
Ly = 40e-9
Lz = Ly
dx = Lx/700
U0 = 0.6*e.value
CFL = 0.99
sigma_x = a/3
N_layer = 110
x0 = N_layer*dx+2*sigma_x
xr = 2*a+2*b+45e-9+dx

m = 1
n = 1

m_eff = 0.023*m_e.value
dt=CFL*2/(2*hbar.value/m_eff*(1/dx**2)+1/hbar.value*U0)
E = 0.3*e.value
print(f'Energy: {E/e.value} eV')
kx = np.sqrt(2*m_eff*E/hbar.value**2)
alpha = 3.0
sigma = alpha * hbar.value / (dt * N_layer)
k = 4 # exponent for the absorbing boundary strength
t_max = 150*Lx*np.sqrt(m_eff/2/E)
# dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)

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

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,CFL=CFL,order=4,ABC=True)
# solver.add_recorder(xr)

# solver.animate(speed=50)
# solver.restart()

# solver.update_loop()
# solver.show_recorder()

###
# With potential barriers:
###

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=4,ABC=True)
# solver.add_barriers(U0)
# solver.add_recorder(xr)
# solver.plot_potential()

# solver.animate(speed = 50)
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

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)
# solver.spectral_source(2)

###
# Validation with analytical solution:
###

def T(order,dt):
    solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
    solver.add_barriers(U0)
    solver.add_recorder(xr)

    solver.update_loop()
    # solver.show_recorder()
    E_num, J_bar = solver.J_freq(np.array(solver.psiRe_record_left),np.array(solver.psiIm_record_left))

    solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
    solver.add_recorder(xr)

    solver.update_loop()
    # solver.show_recorder()
    E_num, J_free = solver.J_freq(np.array(solver.psiRe_record_left),np.array(solver.psiIm_record_left))
    mask=E_num/e.value<0.9

    T_num = np.abs(J_bar[mask]/J_free[mask])
    return E_num[mask]/e.value,T_num

### Analytical solution:

order = 2
dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
solver.add_barriers(U0)
solver.add_recorder(xr)
E_ana,T_ana = solver.analytical_T()
plt.plot(np.real(E_ana)/e.value,T_ana,label='Analytical')

### Numerical solutions:

# order = 2
# dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
# E2,T2 = T(order,dt)
# plt.plot(E2,T2,label='Numerical 2nd order')

order = 4
dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)
E4,T4 = T(order,dt)
plt.plot(E4,T4,label='Numerical 4th order')

plt.xlabel('Energy [eV]')
plt.ylabel('Transmission')
plt.legend()
plt.show()

# ##
# With potential V0
# ##

# V0 = 0.05*e.value

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=4,ABC=True)

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