import numpy as np
import matplotlib.pyplot as plt

class Yee:
    def __init__(self,L,Nx,Ny,CFL,c):
        self.dx = L/Nx
        self.dy = L/Ny
        self.dt = CFL/c/np.sqrt(1/self.dx**2+1/self.dy**2)
        # Fields:
        self.Ez = np.zeros(Nx+1,Ny+1)
        self.Hy = np.zeros(Nx+1,Ny)
        self.Hz = np.zeros(Nx,Ny+1)