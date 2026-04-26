import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from scipy.special import hankel2
from scipy.constants import epsilon_0, mu_0

class Yee:
    def __init__(self,L,Nx,Ny,Nt,dt,N_PML,PML=False,m=4,sigma_max=0.83279,kappa_max=3.221):
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
        self.Jc = np.zeros((Nx+1,Ny+1))
        # Parameters:
        self.eps = np.ones((Nx+1,Ny+1))*epsilon_0
        self.mu = np.ones((Nx+1,Ny+1))*mu_0
        self.c = 1/np.sqrt(np.min(self.eps)*np.min(self.mu))
        self.muy = (self.mu[1:,:]+self.mu[:-1,:])/2
        self.mux = (self.mu[:,1:]+self.mu[:,:-1])/2
        self.sigma = np.zeros((Nx+1,Ny+1))
        # Drude parameters:
        self.drude = False
        self.sigma_DC = np.zeros((Nx+1,Ny+1))
        self.gamma = np.zeros((Nx+1,Ny+1))
        # PML:
        self.PML = PML
        self.sigma_max = sigma_max #(m+1)/(150*np.pi*self.dx[0])
        self.sig = np.array([self.sigma_max*(i/N_PML)**m for i in range(1,N_PML+1)])
        self.Ezx = np.zeros((Nx+1,Ny+1))
        self.Ezy = np.zeros((Nx+1,Ny+1))
        self.sigmy = np.zeros((Nx,Ny+1))
        self.sigmx = np.zeros((Nx+1,Ny))
        self.sigey = np.zeros((Nx+1,Ny+1))
        self.sigex = np.zeros((Nx+1,Ny+1))
        self.kappa_max = kappa_max
        self.kappa = np.array([1+(self.kappa_max-1)*(i/N_PML)**m for i in range(1,N_PML+1)])
        self.kappax = np.ones((Nx+1,Ny+1))
        self.kappay = np.ones((Nx+1,Ny+1))
        
        def put_pml(sig_matrix,kappa_matrix,direction):
            if direction == 'x':
                sig_matrix[:N_PML,:] = self.sig[::-1][:, np.newaxis]
                sig_matrix[-N_PML:,:] = self.sig[:, np.newaxis]
                kappa_matrix[:N_PML,:] = self.kappa[::-1][:, np.newaxis]
                kappa_matrix[-N_PML:,:] = self.kappa[:, np.newaxis]
            if direction == 'y':
                sig_matrix[:,:N_PML] = np.maximum(sig_matrix[:,:N_PML], self.sig[::-1][np.newaxis, :])
                sig_matrix[:,-N_PML:] = np.maximum(sig_matrix[:,-N_PML:], self.sig[np.newaxis, :])
                kappa_matrix[:,:N_PML] = np.maximum(kappa_matrix[:,:N_PML], self.kappa[::-1][np.newaxis, :])
                kappa_matrix[:,-N_PML:] = np.maximum(kappa_matrix[:,-N_PML:], self.kappa[np.newaxis, :])
        
        put_pml(self.sigmy,self.kappay,'y')
        put_pml(self.sigmx,self.kappax,'x')
        put_pml(self.sigey,self.kappay,'y')
        put_pml(self.sigex,self.kappax,'x')
        
        self.make_matrices()
        
    def make_matrices(self):
        self.A = (self.eps/self.dt-self.sigma/2)/(self.eps/self.dt+self.sigma/2)
        self.B = 1/(self.eps/self.dt+self.sigma/2)
        self.Cy = (self.kappax[1:]*self.muy/self.dt-self.sigmy/2)/(self.kappax[1:]*self.muy/self.dt+self.sigmy/2)
        self.Dy = 1/(self.kappax[1:]*self.muy/self.dt+self.sigmy/2)
        self.Cx = (self.kappay[:,1:]*self.mux/self.dt-self.sigmx/2)/(self.kappay[:,1:]*self.mux/self.dt+self.sigmx/2)
        self.Dx = 1/(self.kappay[:,1:]*self.mux/self.dt+self.sigmx/2)
        self.Czy = (self.kappay*self.eps/self.dt-self.sigey/2-self.sigma/2)/(self.kappay*self.eps/self.dt+self.sigey/2+self.sigma/2)
        self.Dzy = 1/(self.kappay*self.eps/self.dt+self.sigey/2+self.sigma/2)
        self.Czx = (self.kappax*self.eps/self.dt-self.sigex/2-self.sigma/2)/(self.kappax*self.eps/self.dt+self.sigex/2+self.sigma/2)
        self.Dzx = 1/(self.kappax*self.eps/self.dt+self.sigex/2+self.sigma/2)
        # For auxiliary differential equation:
        self.A_ade = (self.eps/self.dt - self.sigma_DC/2/(2*self.gamma/self.dt + 1)) / (self.eps/self.dt + self.sigma_DC/2/(2*self.gamma/self.dt + 1))
        self.B_ade = 1/(self.eps/self.dt + self.sigma_DC/2/(2*self.gamma/self.dt + 1))
        self.Czy_ade = (self.kappay*self.eps/self.dt - self.sigey/2 - self.sigma_DC/2/(2*self.gamma/self.dt + 1)) / (self.kappay*self.eps/self.dt + self.sigey/2 + self.sigma_DC/2/(2*self.gamma/self.dt + 1))
        self.Dzy_ade = 1/(self.kappay*self.eps/self.dt + self.sigey/2 + self.sigma_DC/2/(2*self.gamma/self.dt + 1))
        self.Czx_ade = (self.kappax*self.eps/self.dt - self.sigex/2 - self.sigma_DC/2/(2*self.gamma/self.dt + 1)) / (self.kappax*self.eps/self.dt + self.sigex/2 + self.sigma_DC/2/(2*self.gamma/self.dt + 1))
        self.Dzx_ade = 1/(self.kappax*self.eps/self.dt + self.sigex/2 + self.sigma_DC/2/(2*self.gamma/self.dt + 1))
        self.G = (2*self.gamma/self.dt - 1)/(2*self.gamma/self.dt + 1)
        self.F = self.sigma_DC/(2*self.gamma/self.dt + 1)
        self.X = 2*self.gamma/self.dt/(2*self.gamma/self.dt + 1)*self.B_ade
        self.Xy = 2*self.gamma/self.dt/(2*self.gamma/self.dt + 1)*self.Dzy_ade
        self.Xx = 2*self.gamma/self.dt/(2*self.gamma/self.dt + 1)*self.Dzx_ade
        
    def add_source(self,xs,ys,J0,tc,width,Wc=None):
        self.xs = xs
        self.ys = self.Ny-ys
        self.J0 = J0
        self.tc = tc
        self.width = width
        self.Wc = Wc
        self.source_Ez = []
        time = np.arange(0,self.Nt)*self.dt
        if self.Wc==None:
            self.applied_source = self.J0*np.exp(-(time-self.tc)**2/2/self.width**2)
        else:
            self.applied_source = self.J0*np.sin(self.Wc*time)*np.exp(-(time-self.tc)**2/2/self.width**2)
        
    def add_recorder(self,xr,yr):
        self.xr = xr
        self.yr = self.Ny-yr
        self.recorded_Ez = []
        
    def add_material(self,x_start,x_end,y_start,y_end,eps_r,mu_r,sigma):
        if type(x_start) is list:
            for i in range(len(x_start)):
                self.x_start = x_start[i]
                self.x_end = x_end[i]
                self.y_start = y_start[i]
                self.y_end = y_end[i]
                self.eps[x_start[i]:x_end[i],y_start[i]:y_end[i]] *= eps_r[i]
                self.mu[x_start[i]:x_end[i],y_start[i]:y_end[i]] *= mu_r[i]
                self.muy = (self.mu[1:,:]+self.mu[:-1,:])/2
                self.mux = (self.mu[:,1:]+self.mu[:,:-1])/2
                self.sigma[x_start[i]:x_end[i],y_start[i]:y_end[i]] = sigma[i]
        else:
            self.x_start = x_start
            self.x_end = x_end
            self.y_start = y_start
            self.y_end = y_end
            self.eps[x_start:x_end,y_start:y_end] *= eps_r
            self.mu[x_start:x_end,y_start:y_end] *= mu_r
            self.muy = (self.mu[1:,:]+self.mu[:-1,:])/2
            self.mux = (self.mu[:,1:]+self.mu[:,:-1])/2
            self.sigma[x_start:x_end,y_start:y_end] = sigma
        self.make_matrices()

    def add_drude_material(self,x_start,x_end,y_start,y_end,eps_r,sigma_DC,gamma):
        self.x_start = x_start
        self.x_end = x_end
        self.y_start = y_start
        self.y_end = y_end
        self.eps[x_start:x_end,y_start:y_end] *= eps_r
        self.sigma_DC[x_start:x_end,y_start:y_end] = sigma_DC
        self.gamma[x_start:x_end,y_start:y_end] = gamma
        self.drude = True
        self.Ez_old = np.zeros((self.Nx+1,self.Ny+1))
        self.make_matrices()
        
    def restart(self):
        self.Ez = np.zeros((self.Nx+1,self.Ny+1))
        self.Hx = np.zeros((self.Nx+1,self.Ny))
        self.Hy = np.zeros((self.Nx,self.Ny+1))
        self.recorded_Ez = []
        self.n = 0
        self.Ezx = np.zeros((self.Nx+1,self.Ny+1))
        self.Ezy = np.zeros((self.Nx+1,self.Ny+1))
        
    def show_PML(self):
        plt.figure(figsize=(6,5))
        plt.subplot(2,2,1)
        plt.imshow(self.sigmy.T,cmap='viridis',extent=(0,self.L,0,self.L))
        plt.colorbar()
        plt.title('Sigma_y')
        plt.subplot(2,2,2)
        plt.imshow(self.sigmx.T,cmap='viridis',extent=(0,self.L,0,self.L))
        plt.colorbar()
        plt.title('Sigma_x')
        plt.subplot(2,2,3)
        plt.imshow(self.kappax.T,cmap='viridis',extent=(0,self.L,0,self.L))
        plt.colorbar()
        plt.title('Kappa_x')
        plt.subplot(2,2,4)
        plt.imshow(self.kappay.T,cmap='viridis',extent=(0,self.L,0,self.L))
        plt.colorbar()
        plt.title('Kappa_y')
        plt.tight_layout()
        plt.show()
    
    def update(self):
        # Update Ez:
        self.Ez[1:-1,1:-1] = (
            self.A[1:-1,1:-1]*self.Ez[1:-1,1:-1] + 
            self.B[1:-1,1:-1]/self.dx_dual[:, np.newaxis]*(self.Hy[1:,1:-1]-self.Hy[:-1,1:-1])
            - self.B[1:-1,1:-1]/self.dy_dual[np.newaxis,:]*(self.Hx[1:-1,1:]-self.Hx[1:-1,:-1])
        )
        # Source:
        if self.Wc==None:
            self.Ez[self.xs,self.ys] += self.B[self.xs,self.ys]*self.J0*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)
        else:
            self.Ez[self.xs,self.ys] += self.B[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)
        #Update Hy:
        self.Hy[:,1:-1] = self.Hy[:,1:-1] + self.dt/(self.muy[:,1:-1]*self.dx[:,np.newaxis])*(self.Ez[1:,1:-1]-self.Ez[:-1,1:-1])
        #Update Hx:
        self.Hx[1:-1,:] = self.Hx[1:-1,:] - self.dt/(self.mux[1:-1,:]*self.dy[np.newaxis,:])*(self.Ez[1:-1,1:]-self.Ez[1:-1,:-1])
        # Time step:
        self.n += 1
        self.recorded_Ez.append(self.Ez[self.xr,self.yr])
        self.source_Ez.append(self.Ez[self.xs,self.ys])
    
    def update_PML(self):
        # Update Ez:
        self.Ezy[1:-1,1:-1] = (
            self.Czy[1:-1,1:-1]*self.Ezy[1:-1,1:-1] 
            - self.Dzy[1:-1,1:-1]/self.dy_dual[:]*(self.Hx[1:-1,1:]-self.Hx[1:-1,:-1])
        )
        self.Ezx[1:-1,1:-1] = (
            self.Czx[1:-1,1:-1]*self.Ezx[1:-1,1:-1] 
            + self.Dzx[1:-1,1:-1]/self.dx_dual[:, np.newaxis]*(self.Hy[1:,1:-1]-self.Hy[:-1,1:-1])
        )
        # Source:
        if self.Wc==None:
            self.Ezx[self.xs,self.ys] += self.B[self.xs,self.ys]*self.J0*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
            self.Ezy[self.xs,self.ys] += self.B[self.xs,self.ys]*self.J0*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
        else:
            self.Ezx[self.xs,self.ys] += self.B[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
            self.Ezy[self.xs,self.ys] += self.B[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
        self.Ez = self.Ezx + self.Ezy
        #Update Hy:
        self.Hy[:,1:-1] = self.Cy[:,1:-1]*self.Hy[:,1:-1] + self.Dy[:,1:-1]/self.dx[:,np.newaxis]*(self.Ez[1:,1:-1]-self.Ez[:-1,1:-1])
        #Update Hx:
        self.Hx[1:-1,:] = self.Cx[1:-1,:]*self.Hx[1:-1,:] - self.Dx[1:-1,:]/self.dy[np.newaxis,:]*(self.Ez[1:-1,1:]-self.Ez[1:-1,:-1])
        #Time step:
        self.n += 1
        self.recorded_Ez.append(self.Ez[self.xr,self.yr])
        self.source_Ez.append(self.Ez[self.xs,self.ys])

    def update_drude_PML(self):
        # Update Ez:
        self.Ez_old = self.Ez.copy()
        self.Ezy[1:-1,1:-1] = (
            self.Czy_ade[1:-1,1:-1]*self.Ezy[1:-1,1:-1] 
            - self.Dzy_ade[1:-1,1:-1]/self.dy_dual*(self.Hx[1:-1,1:]-self.Hx[1:-1,:-1])
        )
        self.Ezx[1:-1,1:-1] = (
            self.Czx_ade[1:-1,1:-1]*self.Ezx[1:-1,1:-1] 
            + self.Dzx_ade[1:-1,1:-1]/self.dx_dual[:, np.newaxis]*(self.Hy[1:,1:-1]-self.Hy[:-1,1:-1])
        )
        # Source:
        if self.Wc==None:
            self.Ezx[self.xs,self.ys] += self.B_ade[self.xs,self.ys]*self.J0*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
            self.Ezy[self.xs,self.ys] += self.B_ade[self.xs,self.ys]*self.J0*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
        else:
            self.Ezx[self.xs,self.ys] += self.B_ade[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
            self.Ezy[self.xs,self.ys] += self.B_ade[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)/2
        # Induced current
        self.Ezx += self.Xx*self.Jc/2
        self.Ezy += self.Xy*self.Jc/2
        self.Ez = self.Ezx + self.Ezy
        #Update Hy:
        self.Hy[:,1:-1] = self.Cy[:,1:-1]*self.Hy[:,1:-1] + self.Dy[:,1:-1]/self.dx[:,np.newaxis]*(self.Ez[1:,1:-1]-self.Ez[:-1,1:-1])
        #Update Hx:
        self.Hx[1:-1,:] = self.Cx[1:-1,:]*self.Hx[1:-1,:] - self.Dx[1:-1,:]/self.dy[np.newaxis,:]*(self.Ez[1:-1,1:]-self.Ez[1:-1,:-1])
        # Update Jc:
        self.Jc = self.G*self.Jc + self.F*(self.Ez + self.Ez_old)
        #Time step:
        self.n += 1
        self.recorded_Ez.append(self.Ez[self.xr,self.yr])
        self.source_Ez.append(self.Ez[self.xs,self.ys])
    
    def update_drude(self):
        # Update Ez:
        self.Ez_old = self.Ez.copy()
        self.Ez[1:-1,1:-1] = (
            self.A_ade[1:-1,1:-1]*self.Ez[1:-1,1:-1] + 
            self.B_ade[1:-1,1:-1]/self.dx_dual[:, np.newaxis]*(self.Hy[1:,1:-1]-self.Hy[:-1,1:-1])
            - self.B_ade[1:-1,1:-1]/self.dy_dual[np.newaxis,:]*(self.Hx[1:-1,1:]-self.Hx[1:-1,:-1])
        )
        # Source:
        if self.Wc==None:
            self.Ez[self.xs,self.ys] += self.B_ade[self.xs,self.ys]*self.J0*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)
        else:
            self.Ez[self.xs,self.ys] += self.B_ade[self.xs,self.ys]*self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/2/self.width**2)
        # Induced current
        self.Ez += self.X*self.Jc
        #Update Hy:
        self.Hy[:,1:-1] = self.Hy[:,1:-1] + self.dt/(self.muy[:,1:-1]*self.dx[:,np.newaxis])*(self.Ez[1:,1:-1]-self.Ez[:-1,1:-1])
        #Update Hx:
        self.Hx[1:-1,:] = self.Hx[1:-1,:] - self.dt/(self.mux[1:-1,:]*self.dy[np.newaxis,:])*(self.Ez[1:-1,1:]-self.Ez[1:-1,:-1])
        # Update Jc:
        self.Jc = self.G*self.Jc + self.F*(self.Ez + self.Ez_old)
        # Time step:
        self.n += 1
        self.recorded_Ez.append(self.Ez[self.xr,self.yr])
        self.source_Ez.append(self.Ez[self.xs,self.ys])

    def update_loop(self,nt=None):
        if nt is None:
            nt = self.Nt
        for _ in range(nt):
            if self.PML and not self.drude:
                self.update_PML()
            elif self.PML and self.drude:
                self.update_drude_PML()
            elif self.drude:
                self.update_drude()
            else:
                self.update()
                
    def show_Ez(self):
        plt.figure()
        plt.imshow(self.Ez.T,cmap='RdBu_r',extent=(0,self.L,0,self.L),vmin=-1e-5,vmax=1e-5)
        plt.colorbar(label='Ez [V/m]')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Ez field distribution')
        plt.show()
            
    def animate(self,speed=1,repeat=False):
        fig, ax = plt.subplots()
        im = ax.imshow(self.Ez.T,cmap='RdBu_r',extent=(0,self.L,0,self.L),vmin=-0.3,vmax=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0,self.L)
        ax.set_ylim(0,self.L)
        source_marker, = ax.plot(sum(self.dx[:self.xs]), self.L-sum(self.dy[:self.ys]), 'o', color='black', label='source', markersize=2, zorder=3)
        rec1, = ax.plot(sum(self.dx[:self.xr]), self.L-sum(self.dy[:self.yr]), 'x', color='red', label='recorder 1', zorder=3, markersize=6)
        def update(frame):
            self.update_loop(speed)
            im.set_data(self.Ez.T)
            ax.set_title('Ez at t = {:.2f} s'.format(self.n*self.dt))
            return [im,source_marker,rec1]
        
        ani = FuncAnimation(fig, update, frames=self.Nt//speed, interval=int(self.dt * 1000), repeat=repeat)
        # ani.save("simulation.gif", writer="pillow", fps=10)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, label='Ez [V/m]')
        if hasattr(self, 'x_start'):
            x_left = sum(self.dx[:self.x_start])
            x_right = sum(self.dx[:self.x_end])
            width = x_right - x_left
            y_bottom = self.L - sum(self.dy[:self.y_end])
            y_top = self.L - sum(self.dy[:self.y_start])
            height = y_top - y_bottom
            rect = Rectangle((x_left, y_bottom), width, height, linewidth=1, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
        ax.legend(loc='upper left')
        plt.show()
        
    def show_recorder(self):
        plt.figure()
        plt.plot(np.arange(self.n)*self.dt, self.recorded_Ez)
        plt.xlabel('Time [s]')
        plt.ylabel('Ez at recorder [V/m]')
        plt.title(f'Recorded Ez at ({round(sum(self.dx[:self.xr]),2)}, {round(self.L-sum(self.dy[:self.yr]),2)})')
        plt.grid()
        plt.show()

    def analytical_solution(self, plot_all = True, frequency_limit = None):

        if plot_all:
            # Plot time domain response
            plt.plot(self.recorded_Ez, label='Electric field at recorder (V/m)')
            plt.plot(self.applied_source, label='Applied source ($A/m^2$)')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.title('Time domain response')
            plt.show()

        E_freq_sim = np.fft.rfft(self.recorded_Ez)*self.dt
        source_freq = np.fft.rfft(self.applied_source)*self.dt
        omega = 2*np.pi*np.fft.rfftfreq(len(self.recorded_Ez), self.dt)

        # Source and recorder distance
        delta_x = np.sum(self.dx[:self.xs]) - np.sum(self.dx[:self.xr])
        delta_y = np.sum(self.dy[:self.ys]) - np.sum(self.dy[:self.yr])

        # Analytical solution
        E_freq_ana = -self.J0*omega*mu_0/4*hankel2(0, omega/self.c*np.sqrt(delta_x**2+delta_y**2))
        E_freq_ana[0] = 0

        # Restrict to bandwidth of the source
        E_max = np.max(np.abs(source_freq))
        if frequency_limit is None:
            mask = (np.abs(source_freq) > 0.005*E_max)
        else:
            mask = (omega <= frequency_limit)

        if plot_all:
            # Plot frequency domain response
            plt.plot(omega, np.abs(E_freq_sim), label='Electric field at recorder (V/m)')
            plt.plot(omega, np.abs(source_freq), label='Applied source current density ($A/m^2$)')
            plt.xlabel('Frequency (rad/s)')
            plt.legend()
            plt.title('Frequency domain response')
            plt.show()

        if plot_all:
            # Compare with analytical solution
            plt.plot(omega[mask], np.abs(E_freq_sim[mask]/source_freq[mask]*self.Nx*self.Ny/self.L**2),'x', label='Numerical response')
            plt.plot(omega[mask], np.abs(E_freq_ana/self.J0)[mask], label='Analytical response')
            plt.xlabel('Frequency (rad/s)')
            plt.ylabel('|$E_z/J$|')
            plt.legend()
            plt.title('Frequency response comparison')
            plt.show()
        
        return omega[mask],np.abs(E_freq_sim[mask]/source_freq[mask]*self.Nx*self.Ny/np.sum(self.dx)/np.sum(self.dy)), np.abs(E_freq_ana/self.J0)[mask]