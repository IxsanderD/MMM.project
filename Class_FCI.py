import numpy as np
from scipy.sparse import csr_array, diags, eye, kron, lil_array
from scipy.sparse.linalg import splu
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
        self.source_index=[]
        self.eps=eps*np.ones(self.Nx*self.Ny) # Zelfde ordening als velden
        self.mu=mu*np.ones(self.Nx*self.Ny) # Zelfde ordening als velden
        self.sigma=np.zeros(self.Nx*self.Ny)
        self.n=0
        self.kmax=k_max
        self.sigmamax=sigma_max

    def add_source(self,xs,ys,J0,tc,width,Wc):
        self.xs = xs
        self.ys = self.Ny - ys
        self.J0 = J0
        self.tc = tc
        self.width = width
        self.Wc = Wc
        time = np.arange(self.Nt)*self.dt
        # self.applied_source = J0*np.exp(-(time-tc)**2/(2*width**2))
        self.applied_source = J0*np.sin(Wc*time)*np.exp(-(time-tc)**2/(2*width**2))

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
        end=time.perf_counter()

        K_y=np.ravel(K_y)
        K_y_dt=diags(K_y,offsets=0,format='csr')/self.dt
        K_x=np.ravel(K_x)
        K_x_dt=diags(K_x,offsets=0,format='csr')/self.dt
        sigma_x=np.ravel(sigma_x)
        sigma_x_2=diags(sigma_x,offsets=0,format='csr')/2
        sigma_y=np.ravel(sigma_y)
        sigma_y_2=diags(sigma_y,offsets=0,format='csr')/2
        eps=diags(self.eps,offsets=0,format='csr')
        eps_inv=diags(1/self.eps,offsets=0,format='csr')
        mu=diags(self.mu,offsets=0,format='csr')
        sigma=diags(self.sigma,offsets=0,format='csr')
        end=time.perf_counter()

        A_x=kron(np.eye(self.Nx),Ay,format='csr')
        A_y=kron(Ax,np.eye(self.Ny),format='csr')
        A_z=kron(Ax,Ay,format='csr')

        self.M11=csr_array((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.M12=csr_array((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.M21=csr_array((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.M22=csr_array((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.right_matrix=csr_array((6*self.Nx*self.Ny,6*self.Nx*self.Ny))
        
        # block (0,0)
        start=time.perf_counter()
        self.M11[:self.Nx*self.Ny,:self.Nx*self.Ny]=-A_z@(K_y_dt+sigma_y_2@eps_inv)
        self.right_matrix[:self.Nx*self.Ny,:self.Nx*self.Ny]=-A_z@(K_y_dt-sigma_y_2@eps_inv)

        # block (0,1)
        self.M11[:self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z/self.dt
        self.right_matrix[:self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z/self.dt

        # block (1,1)
        self.M11[self.Nx*self.Ny:2*self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z@(eps@K_x_dt+sigma_x_2+sigma/2)
        self.right_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=A_z@(eps@K_x_dt-sigma_x_2-sigma/2)

        # block (1,2)
        self.M11[self.Nx*self.Ny:2*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=kron(Ax,delta_y_inv@Dy,format='csr')/2
        self.right_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=-kron(Ax,delta_y_inv@Dy,format='csr')/2

        # block (1,4)
        self.M12[self.Nx*self.Ny:2*self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=-kron(delta_x_inv@Dx,Ay,format='csr')/2
        self.right_matrix[self.Nx*self.Ny:2*self.Nx*self.Ny,4*self.Nx*self.Ny:5*self.Nx*self.Ny]=kron(delta_x_inv@Dx,Ay,format='csr')/2

        # block (2,2)
        self.M11[2*self.Nx*self.Ny:3*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=-A_x/self.dt
        self.right_matrix[2*self.Nx*self.Ny:3*self.Nx*self.Ny,2*self.Nx*self.Ny:3*self.Nx*self.Ny]=-A_x/self.dt

        # block (2,3)
        self.M12[2*self.Nx*self.Ny:3*self.Nx*self.Ny,:self.Nx*self.Ny]=A_x@(K_x_dt+sigma_x_2@eps_inv)
        self.right_matrix[2*self.Nx*self.Ny:3*self.Nx*self.Ny,3*self.Nx*self.Ny:4*self.Nx*self.Ny]=A_x@(K_x_dt-sigma_x_2@eps_inv)

        # block (3,0)
        self.M21[:self.Nx*self.Ny,:self.Nx*self.Ny]=kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2
        self.right_matrix[3*self.Nx*self.Ny:4*self.Nx*self.Ny,:self.Nx*self.Ny]=-kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2

        # block (3,3)
        self.M22[:self.Nx*self.Ny,:self.Nx*self.Ny]=A_x@(mu@K_y_dt+mu@sigma_y_2@eps_inv)
        self.right_matrix[3*self.Nx*self.Ny:4*self.Nx*self.Ny,3*self.Nx*self.Ny:4*self.Nx*self.Ny]=A_x@(mu@K_y_dt-mu@sigma_y_2@eps_inv)

        # block (4,4)
        self.M22[self.Nx*self.Ny:2*self.Nx*self.Ny,self.Nx*self.Ny:2*self.Nx*self.Ny]=-A_y@(K_x_dt+sigma_x_2@eps_inv)
        self.right_matrix[4*self.Nx*self.Ny:5*self.Nx*self.Ny,4*self.Nx*self.Ny:5*self.Nx*self.Ny]=-A_y@(K_x_dt-sigma_x_2@eps_inv)

        # block (4,5)
        self.M22[self.Nx*self.Ny:2*self.Nx*self.Ny,2*self.Nx*self.Ny:]=A_y@(K_y_dt+sigma_y_2@eps_inv)
        self.right_matrix[4*self.Nx*self.Ny:5*self.Nx*self.Ny,5*self.Nx*self.Ny:]=A_y@(K_y_dt-sigma_y_2@eps_inv)

        # block (5,0)
        self.M21[2*self.Nx*self.Ny:,:self.Nx*self.Ny]=-kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2
        self.right_matrix[5*self.Nx*self.Ny:,:self.Nx*self.Ny]=kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2

        # block (5,5)
        self.M22[2*self.Nx*self.Ny:,2*self.Nx*self.Ny:]=A_y@mu/self.dt
        self.right_matrix[5*self.Nx*self.Ny:,5*self.Nx*self.Ny:]=A_y@mu/self.dt
        end=time.perf_counter()
        print(f"Runtime: {end - start:.6f} seconds")
        
        self.M22_LU=splu(self.M22.tocsc())
        M22_inv_M21=csr_array(self.M22_LU.solve(self.M21.toarray()))
        S=(self.M11-self.M12@M22_inv_M21).tocsc()
        self.S_LU=splu(S)
        end=time.perf_counter()

    def restart(self):
        self.all_fields=np.zeros(6*self.Nx*self.Ny)
        self.n=0
        self.recorded_Ez = []
    
    def update(self):
        b1=self.right_matrix[:3*self.Nx*self.Ny,:]@self.all_fields
        # b1[self.Nx*self.Ny+self.xs*self.Ny+self.ys]+=self.J0*np.exp(-(self.n*self.dt-self.tc)**2/(2*self.width**2))
        b1[self.Nx*self.Ny+self.xs*self.Ny+self.ys]+=self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/(2*self.width**2))
        b2=self.right_matrix[3*self.Nx*self.Ny:,:]@self.all_fields
        M22_inv_b2=self.M22_LU.solve(b2)
        self.all_fields[:3*self.Nx*self.Ny]=self.S_LU.solve(b1-self.M12@M22_inv_b2)
        self.all_fields[3*self.Nx*self.Ny:]=self.M22_LU.solve(b2-self.M21@self.all_fields[:3*self.Nx*self.Ny])
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
        im = ax.imshow(Ez.T,cmap='RdBu_r',extent=(0,np.sum(self.dx),0,np.sum(self.dy)),vmin=-0.05,vmax=0.05)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0,np.sum(self.dx))
        ax.set_ylim(0,np.sum(self.dy))
        ax.set_title('Ez')
        source_marker, = ax.plot(sum(self.dx[:self.xs+1]), np.sum(self.dy) - sum(self.dy[:self.ys+1]), 'o', color='black', label='source', markersize=2, zorder=3)
        rec1, = ax.plot(sum(self.dx[:self.xr+1]), np.sum(self.dy) - sum(self.dy[:self.yr+1]), 'x', color='red', label='recorder 1', zorder=3, markersize=6)
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