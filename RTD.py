import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from Class_RTD import RTD
from astropy.constants.astropyconst20 import m_e,hbar

a = 15
b = 5
Lx = 3*a+2*b
Ly = 40
Lz = Ly
dx = Lx/2000
t_max = 100
dt = 0.7*2/(2*hbar.value/(0.023*m_e.value*dx**2)+0/hbar.value)
x0 = Lx/2
sigma_x = x0/20

kx = 2
sigma = 4e-36
k = 3
N_layer = 80

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

solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True)
solver.animate(m,n)