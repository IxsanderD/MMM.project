import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar,e

a = 15
b = 5
Lx = 3*a+2*b
Ly = 40
Lz = Ly
dx = Lx/3000
t_max = 100
U0 = 0.6*e.value*10**(-18)
dt = 0.7*2/(2*hbar.value/(0.023*m_e.value*dx**2)+U0/hbar.value)
x0 = a/3
sigma_x = a/10

m_eff = 0.023*m_e.value
E = 6*e.value*10**(-18)
kx = np.sqrt(2*m_eff*E/hbar.value**2) # in 1/nm
sigma = 6*E
k = 3 # exponent for the absorbing boundary strength
N_layer = 100

m = 1
n = 1

###
# Without Absorbing Boundaries:
###

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=False)
# solver.animate(m,n)

###
# With Absorbing Boundaries:
###

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True)
# solver.animate(m,n)

###
# With potential barriers:
###

# solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True)
# solver.add_barriers(U0)
# solver.animate(m,n)

###
# Validation with analytical solution:
###

solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True)
solver.add_barriers(U0)

solver.analytical_T()