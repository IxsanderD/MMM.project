import numpy as np
import matplotlib.pyplot as plt
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e,k_B,h

# In this file, the code is given that was used to produce the IV curve.
# The same parameters as in the other files are used.

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

# Function to calculate the transmission numerically
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

# Function to calculate the current for a given voltage
def IV(Vdc,E,T,mu_l=22.436e-3*e.value,Te=0):
    El = mu_l - 6*k_B.value*Te - Vdc
    Er = mu_l + 6*k_B.value*Te
    mask = (El<E)&(Er>E)
    if Te==0:
        # Look at the integrand and integration boundaries
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
        # Look at the integrand and integration boundaries
        # plt.plot(E/e.value,T,label='Transmission')
        # plt.plot(E/e.value,f_L,label='f_L')
        # plt.plot(E/e.value,(1-f_R),label='f_R')
        # plt.plot(El*np.ones(2)/e.value,np.array([np.min(T*(f_L-f_R)),np.max(T*(f_L-f_R))]),label='El')
        # plt.plot(Er*np.ones(2)/e.value,np.array([np.min(T*(f_L-f_R)),np.max(T*(f_L-f_R))]),label='Er')
        # plt.legend()
        # plt.xlabel('Energy [eV]')
        # plt.ylabel('Integrand')
        # plt.show()
        I = 2*e.value/h.value*np.trapezoid(T[mask]*(f_L[mask]-f_R[mask]),E[mask],dx=E[1]-E[0])
        return I

order = 4
colors=['blue','lime','cyan','magenta']

# Calculate the IV curve for different temperatures
for i,Te in enumerate([0,4,77,300]):
    Vdc_values = np.linspace(0,0.1,51)*e.value
    I_values = []
    for Vdc in Vdc_values:
        m = 0
        n = 0
        I = 0
        E, T = numeric_T(order,dt,m,n,Vdc)
        for n in range(1,11):
            for m in range(1,11):
                E_nm = hbar.value**2/(2*0.023*m_e.value)*((np.pi*n/Ly)**2+(np.pi*m/Lz)**2)
                I += IV(Vdc,E+E_nm,T,Te=Te)
        print(f'At T = {Te} K, Vdc={Vdc/e.value:.3f} eV: I={I:.2e} A')
        I_values.append(I)
    I_values = np.array(I_values)
    plt.plot(Vdc_values/e.value*1e3,I_values*1e9,label=f'T={Te} K',color=colors[i])
    # Save the IV curve for later use
    # with open(f'IV_T{Te}.txt','w') as f:
    #     for Vdc,I in zip(Vdc_values,I_values):
    #         f.write(f'{Vdc/e.value:},{I:}\n')

plt.xlabel('Voltage [mV]')
plt.ylabel('Current [nA]')
plt.legend()
plt.show()

# Focus on room temperature and zoom in on the interesting part of the IV curve
Te = 300
Vdc_values = np.linspace(0.025,0.07,91)*e.value
I_values = []
for Vdc in Vdc_values:
    m = 0
    n = 0
    I = 0
    E, T = numeric_T(order,dt,m,n,Vdc)
    for n in range(1,11):
        for m in range(1,11):
            E_nm = hbar.value**2/(2*0.023*m_e.value)*((np.pi*n/Ly)**2+(np.pi*m/Lz)**2)
            I += IV(Vdc,E+E_nm,T,Te=Te)
    print(f'At T = {Te} K, Vdc={Vdc/e.value:.3f} eV: I={I:.2e} A')
    I_values.append(I)

I_values = np.array(I_values)
plt.plot(Vdc_values/e.value*1e3,I_values*1e9,label=f'T={Te} K',color='magenta')
plt.xlabel('Voltage [mV]')
plt.ylabel('Current [nA]')
plt.legend()
plt.show()