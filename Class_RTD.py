import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.constants.astropyconst20 import m_e,hbar,e

class RTD:
    def __init__(self,dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,ABC=True):
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
        self.Energy = hbar.value**2*kx**2/(2*0.023*m_e.value)
        self.Nx = int(self.Lx//self.dx)
        self.Nt = int(self.t_max//self.dt)
        self.C = 1/np.sqrt(np.sqrt(2*np.pi)*self.sigma_x)
        self.psi_Re = np.zeros(self.Nx)
        self.psi_Im = np.zeros(self.Nx)
        self.U = np.zeros(self.Nx)
        self.n = 0
        self.m = 0.023*m_e.value
        self.hbar = hbar.value
        # Absorbing Boundaries:
        self.U_Im = np.zeros(self.Nx)
        if ABC:
            self.U_Im[:N_layer] += np.array([sigma*(i/N_layer)**k for i in range(N_layer-1,-1,-1)])
            self.U_Im[-N_layer:] += np.array([sigma*(i/N_layer)**k for i in range(N_layer)])
        
    def deriv2_2(self,psi):
        res = np.zeros_like(psi)
        res[1:-1] = (psi[2:]-2*psi[1:-1]+psi[:-2])/self.dx**2
        res[0] = (psi[1]-2*psi[0]+psi[-1])/self.dx**2
        res[-1] = (psi[0]-2*psi[-1]+psi[-2])/self.dx**2
        return res
        
    def deriv2_4(self,psi):
        res = np.zeros_like(psi)
        res[2:-2] = (-psi[4:]+16*psi[3:-1]-30*psi[2:-2]+16*psi[1:-3]-psi[:-4])/(12*self.dx**2)
        res[0] = (16*psi[1]-30*psi[0]+16*psi[-1]-psi[-2])/(12*self.dx**2)
        res[1] = (-psi[3]+16*psi[2]-30*psi[1]+16*psi[0]-psi[-1])/(12*self.dx**2)
        res[-2] = (-psi[0]+16*psi[-1]-30*psi[-2]+16*psi[-3]-psi[-4])/(12*self.dx**2)
        res[-1] = (16*psi[0]-30*psi[-1]+16*psi[-2]-psi[-3])/(12*self.dx**2)
        return res
    
    def E(self,m,n):
        return self.hbar**2/(2*self.m)*((np.pi*n/self.Ly)**2+(np.pi*m/self.Lz)**2)
    
    def add_barriers(self,U0):
        self.U[int(self.a//self.dx):int((self.a+self.b)//self.dx)] = U0
        self.U[int((2*self.a+self.b)//self.dx):int((2*self.a+2*self.b)//self.dx)] = U0
        self.Kx = np.sqrt(2*self.m*(self.Energy-U0)/self.hbar**2)
        
    def plot_potential(self):
        plt.plot(np.arange(self.Nx)*self.dx,self.U)
        plt.xlabel('x')
        plt.ylabel('U')
        plt.xlim(0,self.Lx)
        plt.show()
        
    def update_2(self,m,n):
        if self.n==0:
            self.psi_Re = np.array([self.C*np.cos(self.kx*i*self.dx)*np.exp(-(i*self.dx-self.x0)**2/(4*self.sigma_x**2)) for i in range(self.Nx)])
            self.psi_Im = np.array([self.C*np.sin(self.kx*i*self.dx)*np.exp(-(i*self.dx-self.x0)**2/(4*self.sigma_x**2)) for i in range(self.Nx)])
        self.psi_Re += (-self.hbar*self.dt/(2*self.m)*self.deriv2_2(self.psi_Im)
                              + self.dt/self.hbar*(self.U+self.E(m,n))*self.psi_Im
                              - self.dt/self.hbar*self.U_Im*self.psi_Re)
        self.psi_Im += (self.hbar*self.dt/(2*self.m)*self.deriv2_2(self.psi_Re)
                              - self.dt/self.hbar*(self.U+self.E(m,n))*self.psi_Re
                              - self.dt/self.hbar*self.U_Im*self.psi_Im)
        self.n += 1
    
    # def update_4(self,m,n):
    
    
    def update_loop_2(self,m,n):
        for _ in range(self.Nt):
            self.update_2(m,n)
    
    # def update_loop_4(self,m,n):
    #     for _ in range(self.Nt):
    #         self.update_4(m,n)
    
    def animate(self,m,n,speed=1,repeat=False):
        fig, axes = plt.subplots(2,1)
        ax = axes[0]
        ax2 = axes[1]
        im = ax.plot(np.arange(self.Nx)*self.dx,self.psi_Re**2+self.psi_Im**2)[0]
        ax.set_ylabel(r'$|\psi|^2$')
        ax.set_xlim(0,self.Lx)
        ax.set_ylim(0,self.C**2)
        def update(frame):
            self.update_loop_2(m,n)
            im.set_data(np.arange(self.Nx)*self.dx,self.psi_Re**2+self.psi_Im**2)
            ax.set_title('Probability density')
            return [im]
        
        ani = FuncAnimation(fig, update, frames=self.Nt//speed, interval=int(self.dt * 1000), repeat=repeat)
        ax2.plot(np.arange(self.Nx)*self.dx,self.U/e.value*10**18)
        ax2.set_xlabel('x [nm]')
        ax2.set_ylabel('U [eV]')
        ax2.set_xlim(0,self.Lx)
        plt.show()
        
    def show_psi(self):
        plt.plot(np.arange(self.Nx)*self.dx,self.psi_Re,label='Re')
        plt.plot(np.arange(self.Nx)*self.dx,self.psi_Im,label='Im')
        plt.xlabel('x')
        plt.ylabel(r'$\psi$')
        plt.xlim(0,self.Lx)
        plt.legend()
        plt.show()
        
    def analytical_T(self):
        M12 = 1/2*np.array([[1+self.Kx/self.kx,1-self.Kx/self.kx],[1-self.Kx/self.kx,1+self.Kx/self.kx]])
        M23 = 1/2*np.array([[1+self.kx/self.Kx,1-self.kx/self.Kx],[1-self.kx/self.Kx,1+self.kx/self.Kx]])
        M1 = np.array([[np.exp(-1j*self.kx*self.a),0],[0,np.exp(1j*self.kx*self.a)]])
        M2 = np.array([[np.exp(-1j*self.Kx*self.b),0],[0,np.exp(1j*self.Kx*self.b)]])
        M = M12@M2@M23@M1@M12@M2@M23
        print(M)