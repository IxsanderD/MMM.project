import numpy as np
from scipy.sparse import csr_array, diags, eye, kron
from scipy.sparse.linalg import lsqr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time

class FCI:
    def __init__(self,Nx,Ny,Nt,dx,dy,dt,eps,mu,k_max,sigma_max):
        self.Nx=Nx
        self.Ny=Ny
        self.Nt=Nt
        self.dx=dx
        self.dy=dy
        self.dx_dual = np.append((self.dx[1:]+self.dx[:-1])/2,(self.dx[0]+self.dx[-1])/2)
        self.dy_dual = np.append((self.dy[1:]+self.dy[:-1])/2,(self.dy[0]+self.dy[-1])/2)
        self.dt=dt
        self.all_fields=np.zeros(6*Nx*Ny) # Voor ordening, kijk pagina 14 van de cursus
        self.source=np.zeros(6*Nx*Ny)
        self.source_index=[]
        self.eps=eps*np.ones(self.Nx*self.Ny) # Zelfde ordening als velden
        self.mu=mu*np.ones(self.Nx*self.Ny) # Zelfde ordening als velden
        self.sigma=np.zeros(self.Nx*self.Ny)
        self.n=0
        self.kmax=k_max
        self.sigmamax=sigma_max

    def add_source(self,xs,ys,J0,tc,width):
        self.xs = xs
        self.ys = self.Ny - ys
        self.J0 = J0
        self.tc = tc
        self.width = width
        time = np.arange(self.Nt)*self.dt
        self.applied_source = self.J0*np.exp(-(time-self.tc)**2/(2*self.width**2))

    def add_recorder(self,xr,yr):
        self.xr = xr
        self.yr = self.Ny - yr
        self.recorded_Ez = []

    def add_material(self,x_start,x_end,y_start,y_end,eps_r,mu_r,sigma):
        eps=np.reshape(self.eps,(self.Nx,self.Ny))
        eps[x_start:x_end,y_start:y_end]=eps_r*self.eps[0]
        self.eps=np.ravel(eps)

        mu=np.reshape(self.mu,(self.Nx,self.Ny))
        mu[x_start:x_end,y_start:y_end]=mu_r*self.mu[0]
        self.mu=np.ravel(mu)

        sig=np.reshape(self.sigma,(self.Nx,self.Ny))
        sig[x_start:x_end,y_start:y_end]=sigma
        self.sigma=np.ravel(sig)
    
    def construct_update_matrix(self):
        Dx=-diags(np.ones(self.Nx),offsets=0,format='csr')+diags(np.ones(self.Nx-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Nx+1,format='csr')
        Dy=-diags(np.ones(self.Ny),offsets=0,format='csr')+diags(np.ones(self.Ny-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Ny+1,format='csr')
        Ax=1/2*(diags(np.ones(self.Nx),offsets=0,format='csr')+diags(np.ones(self.Nx-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Nx+1,format='csr'))
        Ay=1/2*(diags(np.ones(self.Ny),offsets=0,format='csr')+diags(np.ones(self.Ny-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Ny+1,format='csr'))
        delta_x_inv=diags(1/self.dx,offsets=0,format='csr')
        delta_y_inv=diags(1/self.dy,offsets=0,format='csr')
        k_array=np.array([1+(self.kmax-1)*(i/10)**4 for i in range(1,11)])
        K_y=np.ones((self.Nx,self.Ny))
        K_y[:,self.Ny-10:]=np.tile(k_array,(self.Ny,1))
        K_y[:,:10]=np.tile(np.flip(k_array),(self.Ny,1))
        K_x=np.ones((self.Nx,self.Ny))
        k_array_v=np.reshape(k_array,(10,1))
        K_x[self.Nx-10:,:]=np.repeat(k_array_v,self.Nx,axis=1)
        k_array_v=np.reshape(np.flip(k_array),(10,1))
        K_x[:10,:]=np.repeat(k_array_v,self.Nx,axis=1)
        sigma_array=np.array([self.sigmamax*(i/10)**4 for i in range(1,11)]) # 5/150/np.pi/np.mean(self.dx)
        sigma_y=np.zeros((self.Nx,self.Ny))
        sigma_y[:,self.Ny-10:]=np.tile(sigma_array,(self.Ny,1))
        sigma_y[:,:10]=np.tile(np.flip(sigma_array),(self.Ny,1))
        sigma_array_v=np.reshape(sigma_array,(10,1))
        sigma_x=np.zeros((self.Nx,self.Ny))
        sigma_x[self.Nx-10:,:]=np.repeat(sigma_array_v,self.Nx,axis=1)
        sigma_array_v=np.reshape(np.flip(sigma_array),(10,1))
        sigma_x[:10,:]=np.repeat(sigma_array_v,self.Nx,axis=1)

        K_y=np.ravel(K_y)
        K_y=diags(K_y,offsets=0,format='csr')
        K_x=np.ravel(K_x)
        K_x=diags(K_x,offsets=0,format='csr')
        sigma_x=np.ravel(sigma_x)
        sigma_x=diags(sigma_x,offsets=0,format='csr')
        sigma_y=np.ravel(sigma_y)
        sigma_y=diags(sigma_y,offsets=0,format='csr')
        eps=diags(self.eps,offsets=0,format='csr')
        eps_inv=diags(1/self.eps,offsets=0,format='csr')
        mu=diags(self.mu,offsets=0,format='csr')
        sigma=diags(self.sigma,offsets=0,format='csr')

        A_x=kron(np.eye(self.Nx),Ay,format='csr')
        A_y=kron(Ax,np.eye(self.Ny),format='csr')
        A_z=kron(Ax,Ay,format='csr')

        self.left_matrix=csr_array((6*self.Nx*self.Ny,6*self.Nx*self.Ny))
        self.right_matrix=csr_array((6*self.Nx*self.Ny,6*self.Nx*self.Ny))
        
        # block (0,0)
        self.left_matrix[:self.Nx*self.Ny,:self.Nx*self.Ny]=-A_z@(K_y/self.dt+sigma_y@eps_inv/2)
        self.right_matrix[:self.Nx*self.Ny,:self.Nx*self.Ny]=-A_z@(K_y/self.dt-sigma_y@eps_inv/2)

        # block (0,1)
        self.left_matrix[:self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z/self.dt
        self.right_matrix[:self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z/self.dt

        # block (1,1)
        self.left_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z@(eps@K_x/self.dt+sigma_x/2+sigma/2)
        self.right_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z@(eps@K_x/self.dt-sigma_x/2-sigma/2)

        # block (1,2)
        self.left_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=kron(Ax,delta_y_inv@Dy,format='csr')/2
        self.right_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=-kron(Ax,delta_y_inv@Dy,format='csr')/2

        # block (1,4)
        self.left_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,4*self.Nx*self.Ny:5*self.Nx*self.Ny]=-kron(delta_x_inv@Dx,Ay,format='csr')/2
        self.right_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,4*self.Nx*self.Ny:5*self.Nx*self.Ny]=kron(delta_x_inv@Dx,Ay,format='csr')/2

        # block (2,2)
        self.left_matrix[2*self.Nx*self.Ny:3*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=-A_x/self.dt
        self.right_matrix[2*self.Nx*self.Ny:3*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=-A_x/self.dt

        # block (2,3)
        self.left_matrix[2*self.Nx*self.Ny:3*self.Nx*self.Ny,3*self.Nx*self.Ny:4*self.Nx*self.Ny]=A_x@(K_x/self.dt+sigma_x@eps_inv/2)
        self.right_matrix[2*self.Nx*self.Ny:3*self.Nx*self.Ny,3*self.Nx*self.Ny:4*self.Nx*self.Ny]=A_x@(K_x/self.dt-sigma_x@eps_inv/2)

        # block (3,0)
        self.left_matrix[3*self.Nx*self.Ny:4*self.Nx*self.Ny,:self.Nx*self.Ny]=kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2
        self.right_matrix[3*self.Nx*self.Ny:4*self.Nx*self.Ny,:self.Nx*self.Ny]=-kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2

        # block (3,3)
        self.left_matrix[3*self.Nx*self.Ny:4*self.Nx*self.Ny,3*self.Nx*self.Ny:4*self.Nx*self.Ny]=A_x@(mu@K_y/self.dt+mu@sigma_y@eps_inv/2)
        self.right_matrix[3*self.Nx*self.Ny:4*self.Nx*self.Ny,3*self.Nx*self.Ny:4*self.Nx*self.Ny]=A_x@(mu@K_y/self.dt-mu@sigma_y@eps_inv/2)

        # block (4,4)
        self.left_matrix[4*self.Nx*self.Ny:5*self.Nx*self.Ny,4*self.Nx*self.Ny:5*self.Nx*self.Ny]=-A_y@(K_x/self.dt+sigma_x@eps_inv/2)
        self.right_matrix[4*self.Nx*self.Ny:5*self.Nx*self.Ny,4*self.Nx*self.Ny:5*self.Nx*self.Ny]=-A_y@(K_x/self.dt-sigma_x@eps_inv/2)

        # block (4,5)
        self.left_matrix[4*self.Nx*self.Ny:5*self.Nx*self.Ny,5*self.Nx*self.Ny:]=A_y@(K_y/self.dt+sigma_y@eps_inv/2)
        self.right_matrix[4*self.Nx*self.Ny:5*self.Nx*self.Ny,5*self.Nx*self.Ny:]=A_y@(K_y/self.dt-sigma_y@eps_inv/2)

        # block (5,0)
        self.left_matrix[5*self.Nx*self.Ny:,:self.Nx*self.Ny]=-kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2
        self.right_matrix[5*self.Nx*self.Ny:,:self.Nx*self.Ny]=kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2

        # block (5,5)
        self.left_matrix[5*self.Nx*self.Ny:,5*self.Nx*self.Ny:]=A_y@mu/self.dt
        self.right_matrix[5*self.Nx*self.Ny:,5*self.Nx*self.Ny:]=A_y@mu/self.dt

    def restart(self):
        self.all_fields=np.zeros(6*self.Nx*self.Ny)
        self.n=0
        self.recorded_Ez = []
    
    def update(self):
        self.source[self.Nx*self.Ny+self.xs*self.Ny+self.ys]=self.J0*np.exp(-(self.n*self.dt-self.tc)**2/(2*self.width**2))
        b=self.right_matrix@self.all_fields+self.source
        data=lsqr(self.left_matrix,b)
        self.all_fields=data[0]
        self.n+=1
        Ez=np.reshape(self.all_fields[:self.Nx*self.Ny],(self.Nx,self.Ny))
        self.recorded_Ez.append(Ez[self.xr,self.yr])

    def update_loop(self,nt=None):
        if nt is None:
            nt = self.Nt
        for _ in range(nt):
            self.update()

    def animate(self,speed=1,repeat=False):
        Ez=np.reshape(self.all_fields[:self.Nx*self.Ny],(self.Nx,self.Ny))
        fig, ax = plt.subplots()
        im = ax.imshow(Ez.T,cmap='RdBu_r',extent=(0,np.sum(self.dx),0,np.sum(self.dy)),vmin=-1,vmax=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0,np.sum(self.dx))
        ax.set_ylim(0,np.sum(self.dy))
        ax.set_title('Ez')
        source_marker, = ax.plot(sum(self.dx[:self.xs+1]), self.Ny - sum(self.dy[:self.ys+1]), 'o', color='black', label='source', markersize=2, zorder=3)
        rec1, = ax.plot(sum(self.dx[:self.xr+1]), self.Ny - sum(self.dy[:self.yr+1]), 'x', color='red', label='recorder 1', zorder=3, markersize=6)
        def update(frame):
            self.update_loop(speed)
            Ez=np.reshape(self.all_fields[:self.Nx*self.Ny],(self.Nx,self.Ny))
            im.set_data(Ez.T)
            ax.set_title('Ez at t = {:.2f} s'.format(self.n*self.dt))
            return [im,source_marker, rec1]
        
        ani = FuncAnimation(fig, update, frames=self.Nt//speed, interval=int(self.dt), repeat=repeat)
        # ani.save("simulation.gif", writer="pillow", fps=10)
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