import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

class Yee:
    def __init__(self,L,Nx,Ny,Nt,dt):
        self.L = L
        self.Nx = Nx
        self.Ny = Ny
        self.Nt = Nt
        self.dt = dt
        self.dx = L/Nx*np.ones(Nx)
        self.dy = L/Ny*np.ones(Ny)
        self.dx_dual = (self.dx[1:]+self.dx[:-1])/2
        self.dy_dual = (self.dy[1:]+self.dy[:-1])/2
        self.n = 0 # Timestep
        # Fields:
        self.Ez = np.zeros((Nx+1,Ny+1))
        self.Hx = np.zeros((Nx+1,Ny))
        self.Hy = np.zeros((Nx,Ny+1))
        # Parameters:
        self.eps = np.ones((Nx+1,Ny+1))
        self.mu = np.ones((Nx+1,Ny+1))
        self.c = 1/np.sqrt(np.min(self.eps)*np.min(self.mu))
        self.muy = (self.mu[1:,:]+self.mu[:-1,:])/2
        self.mux = (self.mu[:,1:]+self.mu[:,:-1])/2
        self.sigma = np.zeros((Nx+1,Ny+1))
        self.A = (self.eps/self.dt-self.sigma/2)/(self.eps/self.dt+self.sigma/2)
        self.B = 1/(self.eps/self.dt+self.sigma/2)
        
    def add_source(self,xs,ys,J0,tc,width,Wc):
        self.xs = xs
        self.ys = ys
        self.J0 = J0
        self.tc = tc
        self.width = width
        self.Wc = Wc
        
    def add_recorder(self,xr,yr):
        self.xr = xr
        self.yr = yr
        self.recorded_Ez = []
        
    def restart(self):
        self.Ez = np.zeros((self.Nx+1,self.Ny+1))
        self.Hx = np.zeros((self.Nx+1,self.Ny))
        self.Hy = np.zeros((self.Nx,self.Ny+1))
        self.recorded_Ez = []
        self.n = 0
    
    def update(self):
        # Update Ez:
        self.Ez[1:-1,1:-1] = (
            self.A[1:-1,1:-1]*self.Ez[1:-1,1:-1] + 
            self.B[1:-1,1:-1]/self.dx_dual[:, np.newaxis]*(self.Hy[1:,1:-1]-self.Hy[:-1,1:-1])
            - self.B[1:-1,1:-1]/self.dy_dual[:]*(self.Hx[1:-1,1:]-self.Hx[1:-1,:-1])
        )
        # Source:
        self.Ez[self.xs,self.ys] += -self.B[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)
        #Update Hy:
        self.Hy[:,1:-1] = self.Hy[:,1:-1] + self.dt/(self.muy[:,1:-1]/self.dx[:,np.newaxis])*(self.Ez[1:,1:-1]-self.Ez[:-1,1:-1])
        #Update Hx:
        self.Hx[1:-1,:] = self.Hx[1:-1,:] - self.dt/(self.mux[1:-1,:]/self.dy[np.newaxis,:])*(self.Ez[1:-1,1:]-self.Ez[1:-1,:-1])
        self.n += 1
        self.recorded_Ez.append(self.Ez[self.xr,self.yr])
    
    def update_loop(self,nt=None):
        if nt is None:
            nt = self.Nt
        for _ in range(nt):
            self.update()
            
    def animate(self,speed=1,repeat=False):
        fig, ax = plt.subplots()
        im = ax.imshow(self.Ez.T,cmap='RdBu_r',extent=(0,self.L,0,self.L),vmin=-0.1,vmax=0.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0,self.L)
        ax.set_ylim(0,self.L)
        ax.set_title('Ez')
        source_marker, = ax.plot(sum(self.dx[:self.xs]), sum(self.dy[:self.ys]), 'o', color='black', label='source', markersize=2, zorder=3)
        rec1, = ax.plot(sum(self.dx[:self.xr]), sum(self.dy[:self.yr]), 'x', color='red', label='recorder 1', zorder=3, markersize=6)
        def update(frame):
            self.update_loop(speed)
            im.set_data(self.Ez.T)
            return [im,source_marker,rec1]
        
        ani = FuncAnimation(fig, update, frames=self.Nt//speed, interval=int(self.dt * 1000), repeat=repeat)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, label='Ez [V/m]')
        ax.legend(loc='upper left')
        plt.show()
        
    def show_recorder(self):
        plt.figure()
        plt.plot(np.arange(self.n)*self.dt, self.recorded_Ez)
        plt.xlabel('Time [s]')
        plt.ylabel('Ez at recorder [V/m]')
        plt.title(f'Recorded Ez at ({round(sum(self.dx[:self.xr]),1)}, {round(sum(self.dy[:self.yr]),1)})')
        plt.grid()
        plt.show()