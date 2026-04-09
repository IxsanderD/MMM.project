import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from Class_RTD import RTD

a = 15
b = 5
Lx = 3*a+2*b
Ly = 40
Lz = Ly
dx = Lx//500
t_max = 1
dt = t_max//100
x0 = a/3
sigma_x = x0/5
kx = 1

solver = RTD(dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx)