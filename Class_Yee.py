import numpy as np
import matplotlib.pyplot as plt

class Yee:
    def __init__(self,L,Nx,Ny,Nt,dt):
        self.dx = L/Nx*np.ones(Nx)
        self.dy = L/Ny*np.ones(Ny)
        self.dx_dual = (self.dx[1:]+self.dx[:-1])/2
        self.dy_dual = (self.dy[1:]+self.dy[:-1])/2
        self.dt = dt
        self.Nt = Nt
        self.n = 0 # Timestep
        # Fields:
        self.Ez = np.zeros((Nx+1,Ny+1))
        self.Hx = np.zeros((Nx+1,Ny))
        self.Hy = np.zeros((Nx,Ny+1))
        # Parameters:
        self.eps = np.ones((Nx+1,Ny+1))
        self.mu = np.ones((Nx+1,Ny+1))
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
    
    def update(self):
        # Update Ez:
        self.Ez[1:-1,1:-1] = (
            self.A[1:-1,1:-1]*self.Ez[1:-1,1:-1]+self.B[1:-1,1:-1]/self.dx_dual[:]*(self.Hy[1:,1:-1]-self.Hy[:-1,1:-1])
            - self.B[1:-1,1:-1]/self.dy_dual[:]*(self.Hx[1:-1,1:]-self.Hx[1:-1,:-1])
        )
        # Source:
        self.Ez[self.xs,self.ys] += -self.B[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)
        #Update Hy:
        self.Hy[:,1:-1] = self.Hy[:,1:-1] + self.dt/(self.muy[:,1:-1]/self.dx[:,np.newaxis])*(self.Ez[1:,1:-1]-self.Ez[:-1,1:-1])
        #Update Hx:
        self.Hx[1:-1,:] = self.Hx[1:-1,:] - self.dt/(self.mux[1:-1,:]/self.dy[np.newaxis,:])*(self.Ez[1:-1,1:]-self.Ez[1:-1,:-1])
        self.n += 1
    
    def update_loop(self,nt=None):
        if nt is None:
            nt = self.Nt
        for _ in range(nt):
            self.update()
            
    