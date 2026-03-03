import numpy as np
import matplotlib.pyplot as plt
from Classes import Yee

L = 1
Nx = 100
Ny = 100
CFL = 1
c = 1

solver = Yee(L,Nx,Ny,CFL,c)