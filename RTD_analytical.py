import numpy as np
import matplotlib.pyplot as plt
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e

a = 15e-9
b = 5e-9
Lx = (3*a+2*b+78e-9) # Extra space for barrier to not have an influence
Ly = 40e-9
Lz = Ly
dx = Lx/800
U0 = 0.6*e.value
CFL = 1.00
sigma_x = a/3
N_layer = 140
x0 = N_layer*dx+2*sigma_x
xr = 2*a+2*b+48e-9+dx

m = 0
n = 0

m_eff = 0.023*m_e.value
dt=CFL*2/(2*hbar.value/m_eff*(1/dx**2)+1/hbar.value*U0)
E = 0.3*e.value
print(f'Energy: {E/e.value} eV')
kx = np.sqrt(2*m_eff*E/hbar.value**2)
alpha = 3.0
sigma = alpha * hbar.value / (dt * N_layer)
k = 4 # exponent for the absorbing boundary strength
t_max = 250*Lx*np.sqrt(m_eff/2/E)

print(f't_max: {t_max}')

def numeric_T(order,dt,m,n):
    solver = RTD(dx,dt,a,b,Lx,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
    solver.add_barriers(U0)
    solver.add_recorder(xr)

    solver.update_loop()
    E_num, J_bar = solver.J_freq()

    solver = RTD(dx,dt,a,b,Lx,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
    solver.add_recorder(xr)

    solver.update_loop()
    E_num, J_free = solver.J_freq()
    mask=E_num/e.value<0.9

    T_num = np.abs(J_bar[mask]/J_free[mask])
    return E_num[mask]/e.value,T_num

###
# Validation with analytical solution:
###

### Analytical solution:

order = 2
dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
solver = RTD(dx,dt,a,b,Lx,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
solver.add_barriers(U0)
solver.add_recorder(xr)
E_ana,T_ana = solver.analytical_T()
plt.plot(np.real(E_ana)/e.value,T_ana,label='Analytical',color='mediumturquoise',linestyle='solid',zorder=1)

### Numerical solutions:

order = 2
dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
E2,T2 = numeric_T(order,dt,m,n)
plt.plot(E2,T2,label='Numerical 2nd order',color='lime',linestyle='dashdot',zorder=10)

order = 4
dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)
E4,T4 = numeric_T(order,dt,m,n)
plt.plot(E4,T4,label='Numerical 4th order',color='blue',linestyle='dashed',zorder=10)

plt.xlabel('Energy [eV]')
plt.ylabel('Transmission')
plt.grid()
plt.legend()
plt.show()

# We can even go a step further and compare both orders:

### Comparison of orders:

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