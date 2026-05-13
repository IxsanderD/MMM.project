import numpy as np
import matplotlib.pyplot as plt
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e,k_B,h

### Functioons:

def numeric_T(order,dt,m,n,V0=0):
    solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order=order,ABC=True)
    solver.add_barriers(U0)
    solver.add_potential(V0)
    solver.add_recorder(xr)

    solver.update_loop()
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
        I = 2*e.value/h.value*np.trapezoid(T[mask],E[mask],dx=E[1]-E[0])
        return I
    else:
        f_L = 1/(np.exp((E-mu_l)/k_B.value/Te)+1)
        f_R = 1/(np.exp((E-mu_l+Vdc)/k_B.value/Te)+1)
        I = 2*e.value/h.value*np.trapezoid(T[mask]*(f_L[mask]-f_R[mask]),E[mask],dx=E[1]-E[0])
        return I
    
### Parameters:

a = 15e-9
b = 5e-9
Lx = (3*a+2*b+108e-9) # Extra space for barrier to not have an influence
Ly = 40e-9
Lz = Ly
dx = Lx/900
U0 = 0.6*e.value
CFL = 0.9
sigma_x = a/3
N_layer = 150
x0 = N_layer*dx+2*sigma_x
xr = 3*a+3*b+48e-9+dx

m = 0
n = 0

m_eff = 0.023*m_e.value
dt=CFL*2/(2*hbar.value/m_eff*(1/dx**2)+1/hbar.value*U0)
E = 0.3*e.value
kx = np.sqrt(2*m_eff*E/hbar.value**2)
alpha = 3.0
sigma = alpha * hbar.value / (dt * N_layer)
k = 4
t_max = 30*Lx*np.sqrt(m_eff/2/E)

### Experiment: (change barrier distance)

E, T = numeric_T(2,dt,m,n)
plt.plot(E/e.value,T,label=f'Numerical 2th order')

dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)
E, T = numeric_T(4,dt,m,n)
plt.plot(E/e.value,T,label=f'Numerical 4th order')

plt.xlabel('Energy [eV]')
plt.ylabel('Transmission')
plt.legend()
plt.show()

# Te = 0
# order = 4
# dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2)+U0/hbar.value)
# Vdc_values = np.linspace(0,0.1,11)*e.value
# I_values = []

# plt.figure(1)
# for Vdc in Vdc_values:
#     m = 0
#     n = 0
#     I = 0
#     E, T = numeric_T(order,dt,m,n,Vdc)
#     plt.plot(E/e.value,T,label=f'Vdc={Vdc/e.value:.2f} eV')
#     for n in range(1,10):
#         for m in range(1,10):
#             E_nm = hbar.value**2/(2*0.023*m_e.value)*((np.pi*n/Ly)**2+(np.pi*m/Lz)**2)
#             I += IV(Vdc,E+E_nm,T,Te=Te)
#     print(f'At Vdc={Vdc/e.value:.2f} eV, I={I:.2e} A')
#     I_values.append(I)
# I_values = np.array(I_values)

# plt.xlabel('Energy [eV]')
# plt.ylabel('Transmission')
# plt.legend()

# plt.figure(2)
# plt.plot(Vdc_values/e.value,I_values*10**12)
# plt.xlabel('Voltage [V]')
# plt.ylabel('Current [pA]')
# plt.show()