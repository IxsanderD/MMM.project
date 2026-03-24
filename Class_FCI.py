import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import spsolve, svds
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

class FCI:
    def __init__(self,Nx,Ny,Nt,dx,dy,dt,eps,mu):
        self.Nx=Nx
        self.Ny=Ny
        self.Nt=Nt
        self.dx=dx
        self.dy=dy
        self.dx_dual = np.append((self.dx[1:]+self.dx[:-1])/2,(self.dx[0]+self.dx[-1])/2)
        self.dy_dual = np.append((self.dy[1:]+self.dy[:-1])/2,(self.dy[0]+self.dy[-1])/2)
        self.dt=dt
        self.all_fields=np.zeros(3*Nx*Ny) # Voor ordening, kijk pagina 14 van de cursus
        self.source=np.zeros(3*Nx*Ny)
        self.source_index=[]
        self.eps=eps # Zelfde ordening als velden
        self.mu=mu # Zelfde ordening als velden
        self.n=0

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
    
    def construct_update_matrix(self):
        Dx=-diags(np.ones(self.Nx),offsets=0,format='csr')+diags(np.ones(self.Nx-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Nx+1,format='csr')
        Dy=-diags(np.ones(self.Ny),offsets=0,format='csr')+diags(np.ones(self.Ny-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Ny+1,format='csr')
        Ax=1/2*(diags(np.ones(self.Nx),offsets=0,format='csr')+diags(np.ones(self.Nx-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Nx+1,format='csr'))
        Ay=1/2*(diags(np.ones(self.Ny),offsets=0,format='csr')+diags(np.ones(self.Ny-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Ny+1,format='csr'))
        delta_x_inv=diags(1/self.dx,offsets=0,format='csr')
        delta_y_inv=diags(1/self.dy,offsets=0,format='csr')
        # delta_x_dual=diags(self.dx_dual,offsets=0,format='csr')
        # delta_y_dual=diags(self.dy_dual,offsets=0,format='csr')

        K_x=np.ones(self.Nx*self.Ny)
        K_y=np.ones(self.Nx*self.Ny)
        sigma_x=np.zeros(self.Nx*self.Ny)
        sigma_y=np.zeros(self.Nx*self.Ny)

        # delta_dual=csr_matrix((2*self.Nx*self.Ny,2*self.Nx*self.Ny))
        # delta_dual[:self.Nx*self.Ny,:self.Nx*self.Ny]=kron(delta_x_dual,np.eye(self.Ny),format='csr')
        # delta_dual[self.Nx*self.Ny:,self.Nx*self.Ny:]=kron(np.eye(self.Nx),delta_y_dual,format='csr')

        # Als we anisotrope media gaan bestuderen hebben we deze nodig, maar nu nog niet
        # delta_dual_inv=np.zeros((2*self.Nx*self.Ny,2*self.Nx*self.Ny))
        # delta_dual_inv[diags_indices_from(delta_dual_inv)]=1/delta_dual[diags_indices_from(delta_dual)]

        star_c_o_curl_10=csr_matrix((2*self.Nx*self.Ny,self.Nx*self.Ny))
        star_c_o_curl_10[:self.Nx*self.Ny,:]=kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')
        star_c_o_curl_10[self.Nx*self.Ny:,:]=kron(-delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')

        star_c_o_curl_01=csr_matrix((self.Nx*self.Ny,2*self.Nx*self.Ny))
        star_c_o_curl_01[:,:self.Nx*self.Ny]=kron(-Ax,delta_y_inv@Dy,format='csr')
        star_c_o_curl_01[:,self.Nx*self.Ny:]=kron(delta_x_inv@Dx,Ay,format='csr')
        A_00=kron(Ax,Ay,format='csr')

        A_11=csr_matrix((2*self.Nx*self.Ny,2*self.Nx*self.Ny))
        A_11[:self.Nx*self.Ny,:self.Nx*self.Ny]=kron(np.eye(self.Nx),Ay,format='csr')
        A_11[self.Nx*self.Ny:,self.Nx*self.Ny:]=kron(Ax,np.eye(self.Ny),format='csr')

        eps=diags(self.eps,offsets=0,format='csr')
        A_eps_dt=1/self.dt*A_00@eps # Hier hebben we normaal delta_duals nodig, maar negeren we voorlopig
        
        mu=csr_matrix((2*self.Nx*self.Ny,2*self.Ny*self.Ny))
        mu[[i for i in range(self.Nx*self.Ny)],[i for i in range(self.Nx*self.Ny)]]=self.mu
        mu[[i+self.Nx*self.Ny for i in range(self.Nx*self.Ny)],[i+self.Nx*self.Ny for i in range(self.Nx*self.Ny)]]=self.mu
        A_mu_dt=1/self.dt*A_11@mu # Hier hebben we normaal delta_duals nodig, maar negeren we voorlopig
 
        self.left_matrix=csr_matrix((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.left_matrix[:self.Nx*self.Ny,:self.Nx*self.Ny]=A_eps_dt
        self.left_matrix[:self.Nx*self.Ny,self.Nx*self.Ny:]=-star_c_o_curl_01/2
        self.left_matrix[self.Nx*self.Ny:,:self.Nx*self.Ny]=star_c_o_curl_10/2
        self.left_matrix[self.Nx*self.Ny:,self.Nx*self.Ny:]=A_mu_dt

        self.right_matrix=csr_matrix((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.right_matrix[:self.Nx*self.Ny,:self.Nx*self.Ny]=A_eps_dt
        self.right_matrix[:self.Nx*self.Ny,self.Nx*self.Ny:]=star_c_o_curl_01/2
        self.right_matrix[self.Nx*self.Ny:,:self.Nx*self.Ny]=-star_c_o_curl_10/2
        self.right_matrix[self.Nx*self.Ny:,self.Nx*self.Ny:]=A_mu_dt

        self.left_matrix[0,:]=np.where(np.array([i for i in range(3*self.Nx*self.Ny)])==0,1,0) # Moeten velden ergens pinnen zodat matrix niet singulier wordt
        self.right_matrix[0,:]=np.where(np.array([i for i in range(3*self.Nx*self.Ny)])==0,1,0)
        self.left_matrix[self.Nx*self.Ny,:]=np.where(np.array([i for i in range(3*self.Nx*self.Ny)])==self.Nx*self.Ny,1,0)
        self.right_matrix[self.Nx*self.Ny,:]=np.where(np.array([i for i in range(3*self.Nx*self.Ny)])==self.Nx*self.Ny,1,0)
        self.left_matrix[2*self.Nx*self.Ny,:]=np.where(np.array([i for i in range(3*self.Nx*self.Ny)])==2*self.Nx*self.Ny,1,0)
        self.right_matrix[2*self.Nx*self.Ny,:]=np.where(np.array([i for i in range(3*self.Nx*self.Ny)])==2*self.Nx*self.Ny,1,0)

        self.left_matrix=self.left_matrix.tocsr()
        self.right_matrix=self.right_matrix.tocsr()

    def restart(self):
        self.all_fields=np.zeros(3*self.Nx*self.Ny)
        self.n=0
    
    def update(self):
        self.source[self.xs*self.Ny+self.ys]=self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/(2*self.width**2))
        self.all_fields=spsolve(self.left_matrix,self.right_matrix@self.all_fields+self.source)
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
        source_marker, = ax.plot(sum(self.dx[:self.xs]), sum(self.dy[:self.ys]), 'o', color='black', label='source', markersize=2, zorder=3)
        def update(frame):
            self.update_loop(speed)
            Ez=np.reshape(self.all_fields[:self.Nx*self.Ny],(self.Nx,self.Ny))
            im.set_data(Ez.T)
            return [im,source_marker]
        
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


