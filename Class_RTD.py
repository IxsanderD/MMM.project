import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.constants.astropyconst20 import m_e,hbar

class RTD:
    def __init__(self,dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx):
        self.dx = dx
        self.dt = dt
        self.a = a
        self.b = b
        self.Lx = 3*a+2*b
        self.Ly = Ly
        self.Lz = Lz
        self.t_max = t_max
        self.x0 = x0
        self.sigma_x = sigma_x
        self.kx = kx
        # self.C = 
        self.psi_Re = np.zeros(self.Lx/dx)
        self.psi_Im = np.zeros(self.Lx/dx)
        self.U = np.zeros(self.Lx/dx)
        self.n = 0
        self.m = 0.023*m_e.value
        self.hbar = hbar.value
        
    def deriv2_2(self,psi):
        return (psi[2:]-2*psi[1:-1]+psi[:-2])/self.dx**2
        
    def deriv2_4(self,psi):
        return (-psi[4:]+16*psi[3:-1]-30*psi[2:-2]+16*psi[1:-3]-psi[:-4])/(12*self.dx**2)
    
    def E(self,m,n):
        return self.hbar**2/(2*self.m)*((np.pi*n/self.Ly)**2+(np.pi*m/self.Lz)**2)
        
    def update_2(self,m,n):
        if n==0:
            self.psi_Re = np.array([])
        self.psi_Re[1:-1] += (-self.hbar*self.dt/(2*self.m)*self.deriv2_2(self.psi_Im)
                              + self.dt/self.hbar*(self.U[1:-1]+self.E(m,n))*self.psi_Im[1:-1])
        self.psi_Im[1:-1] += (self.hbar*self.dt/(2*self.m)*self.deriv2_2(self.psi_Re)
                              - self.dt/self.hbar*(self.U[1:-1]+self.E(m,n))*self.psi_Re[1:-1])
        self.n += 1
    
    # def update_4(self,m,n):
    
    
    def update_loop_2(self,m,n):
        for _ in range(self.t_max//self.dt):
            self.update_2(m,n)
    
    # def update_loop_4(self,m,n):
    #     for _ in range(self.t_max//self.dt):
    #         self.update_4(m,n)