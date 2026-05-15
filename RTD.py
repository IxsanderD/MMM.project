import numpy as np
import matplotlib.pyplot as plt
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e,k_B,h

# In this file, we will peform all of our experiments. We start by looking at some animations and plots of our potentials to get our first
# visual check of the code's validity. We do this by defining the parameters of our class.

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
t_max = 10*Lx*np.sqrt(m_eff/2/E)
# dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)

print(f't_max: {t_max}')

# Now, we make some aniamtions and plots:

###
# Without Absorbing Boundaries:
###

# solver = RTD(dx,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=False)
# solver.add_recorder(xr)
# solver.animate(speed = 1000)

###
# With Absorbing Boundaries:
###

solver = RTD(dx,dt,a,b,Lx,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,CFL=CFL,order=4,ABC=True)
solver.add_recorder(xr)

solver.animate(speed=200)
solver.restart()

# solver.update_loop()
# solver.show_recorder()

###
# With potential barriers:
###

solver = RTD(dx,dt,a,b,Lx,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=4,ABC=True)
solver.add_barriers(U0)
solver.add_recorder(xr)
solver.plot_potential()

solver.animate(speed = 200)
solver.restart()

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

# All of these results look fine. We thus start with our first real validation, namely comparing the analytical solution to the numeric
# one. We do this for the 2nd and 4th order scheme. We again define all of our parameters

a = 15e-9
b = 5e-9
Lx = (3*a+2*b+108e-9) # Extra space for barrier to not have an influence
Ly = 40e-9
Lz = Ly
dx = Lx/600
U0 = 0.6*e.value
CFL = 0.9
sigma_x = a/3
N_layer = 100
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
t_max = 30*Lx*np.sqrt(m_eff/2/E)

print(f't_max: {t_max}')

# a = 15e-9
# b = 5e-9
# Lx = (3*a+2*b+108e-9) # Extra space for barrier to not have an influence
# Ly = 40e-9
# Lz = Ly
# dx = Lx/1500
# U0 = 0.6*e.value
# CFL = 0.99
# sigma_x = a/3
# N_layer = 200
# x0 = N_layer*dx+2*sigma_x
# xr = 2*a+2*b+48e-9+dx

# m = 0
# n = 0

# m_eff = 0.023*m_e.value
# dt=CFL*2/(2*hbar.value/m_eff*(1/dx**2)+1/hbar.value*U0)
# E = 0.3*e.value
# print(f'Energy: {E/e.value} eV')
# kx = np.sqrt(2*m_eff*E/hbar.value**2)
# alpha = 1.0
# sigma = alpha * hbar.value / (dt * N_layer)
# k = 4 # exponent for the absorbing boundary strength
# t_max = 50*Lx*np.sqrt(m_eff/2/E)
# # dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
# dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)

# print(f't_max: {t_max}')

# ##
# With potential V0
# ##

# V0 = 0.05*e.value

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=2,ABC=True)

# solver.add_barriers(U0)
# solver.add_potential(V0)
# solver.plot_potential()
# solver.add_recorder(xr)
# solver.animate(speed = 200)
# solver.restart()

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=4,ABC=True)

# solver.add_barriers(U0)
# solver.add_potential(V0)
# solver.plot_potential()
# solver.add_recorder(xr)
# order = 4
# dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)
# E4,T4 = numeric_T(order,dt,m,n)
# plt.plot(E4,T4,label='Numerical 4th order')
# plt.xlabel('Energy [eV]')
# plt.ylabel('Transmission')
# plt.legend()
# plt.show()

###
# IV curve:
###
a = 15e-9
b = 5e-9
Lx = (3*a+2*b+78e-9) # Extra space for barrier to not have an influence
Ly = 40e-9
Lz = Ly
dx = Lx/500
U0 = 0.6*e.value
CFL = 0.99
sigma_x = a/3
N_layer = 70
x0 = N_layer*dx+2*sigma_x
xr = 2*a+2*b+48e-9+dx

m = 0
n = 0

m_eff = 0.023*m_e.value
dt=CFL*2/(8*hbar.value/3/m_eff*(1/dx**2)+1/hbar.value*U0)
E = 0.3*e.value
print(f'Energy: {E/e.value} eV')
kx = np.sqrt(2*m_eff*E/hbar.value**2)
alpha = 3.0
sigma = alpha * hbar.value / (dt * N_layer)
k = 4 # exponent for the absorbing boundary strength
t_max = 150*Lx*np.sqrt(m_eff/2/E)

def numeric_T(order,dt,m,n,V0=0):
    solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
    solver.add_barriers(U0)
    solver.add_potential(V0)
    solver.add_recorder(xr)

    solver.update_loop()
    # solver.show_recorder()
    E_num, J_bar = solver.J_freq()

    solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
    solver.add_potential(V0)
    solver.add_recorder(xr)

    solver.update_loop()
    E_num, J_free = solver.J_freq()
    mask=E_num/e.value<0.9

    T_num = np.abs(J_bar[mask]/J_free[mask])
    return E_num[mask],T_num

def IV(Vdc,E,T,mu_l=22.436e-3*e.value,Te=0):
    El = mu_l - 6*k_B.value*Te - Vdc
    Er = mu_l + 6*k_B.value*Te
    mask = (El<E)&(Er>E)
    if Te==0:
        # plt.plot(E/e.value,T,label='Transmission')
        # plt.plot(El*np.ones(2)/e.value,np.array([np.min(T),np.max(T)]),label='El')
        # plt.plot(Er*np.ones(2)/e.value,np.array([np.min(T),np.max(T)]),label='Er')
        # plt.legend()
        # plt.xlabel('Energy [eV]')
        # plt.ylabel('Integrand')
        # plt.show()
        I = 2*e.value/h.value*np.trapezoid(T[mask],E[mask],dx=E[1]-E[0])
        return I
    else:
        f_L = 1/(np.exp((E-mu_l)/k_B.value/Te)+1)
        f_R = 1/(np.exp((E-mu_l+Vdc)/k_B.value/Te)+1)
        # plt.plot(E/e.value,T,label='Transmission')
        # plt.plot(E/e.value,f_L,label='f_L')
        # plt.plot(E/e.value,f_R,label='f_R')
        # plt.plot(El*np.ones(2)/e.value,np.array([np.min(T*(f_L-f_R)),np.max(T*(f_L-f_R))]),label='El')
        # plt.plot(Er*np.ones(2)/e.value,np.array([np.min(T*(f_L-f_R)),np.max(T*(f_L-f_R))]),label='Er')
        # plt.legend()
        # plt.xlabel('Energy [eV]')
        # plt.ylabel('Integrand')
        # plt.show()
        I = 2*e.value/h.value*np.trapezoid(T[mask]*(f_L[mask]-f_R[mask]),E[mask],dx=E[1]-E[0])
        return I

Te = 0
order = 4
dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)
Vdc_values = np.linspace(0,0.1,11)*e.value
I_values = []


# for Te in [0,4,77,300]:
#     Vdc_values = np.linspace(0,0.1,51)*e.value
#     I_values = []
#     for Vdc in Vdc_values:
#         m = 0
#         n = 0
#         I = 0
#         E, T = numeric_T(order,dt,m,n,Vdc)
#         # plt.plot(E/e.value,T,label=f'Vdc={Vdc/e.value:.2f} eV')
#         for n in range(1,11):
#             for m in range(1,11):
#                 E_nm = hbar.value**2/(2*0.023*m_e.value)*((np.pi*n/Ly)**2+(np.pi*m/Lz)**2)
#                 I += IV(Vdc,E+E_nm,T,Te=Te)
#         print(f'At T = {Te} K, Vdc={Vdc/e.value:.3f} eV: I={I:.2e} A')
#         I_values.append(I)
#     I_values = np.array(I_values)
#     plt.plot(Vdc_values/e.value,I_values*10**12,label=f'T={Te} K')
#     with open(f'IV_T{Te}.txt','w') as f:
#         for Vdc,I in zip(Vdc_values,I_values):
#             f.write(f'{Vdc/e.value:},{I:}\n')

# # plt.xlabel('Energy [eV]')
# # plt.ylabel('Transmission')
# # plt.legend()

# plt.xlabel('Voltage [V]')
# plt.ylabel('Current [pA]')
# plt.legend()
# plt.show()

Te = 300
Vdc_values = np.linspace(0.025,0.07,91)*e.value
I_values = []
for Vdc in Vdc_values:
    m = 0
    n = 0
    I = 0
    E, T = numeric_T(order,dt,m,n,Vdc)
    # plt.plot(E/e.value,T,label=f'Vdc={Vdc/e.value:.2f} eV')
    for n in range(1,11):
        for m in range(1,11):
            E_nm = hbar.value**2/(2*0.023*m_e.value)*((np.pi*n/Ly)**2+(np.pi*m/Lz)**2)
            I += IV(Vdc,E+E_nm,T,Te=Te)
    print(f'At T = {Te} K, Vdc={Vdc/e.value:.3f} eV: I={I:.2e} A')
    I_values.append(I)
I_values = np.array(I_values)
plt.plot(Vdc_values/e.value*1e3,I_values*10**9,label=f'T={Te} K')
plt.xlabel('Voltage [mV]')
plt.ylabel('Current [nA]')
plt.legend()
plt.show()