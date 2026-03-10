import numpy as np
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
    
    def construct_update_matrix(self):
        Dx=-np.diag(np.ones(self.Nx),k=0)+np.diag(np.ones(self.Nx-1),k=1)+np.diag(np.ones(1),k=-self.Nx+1)
        Dy=-np.diag(np.ones(self.Ny),k=0)+np.diag(np.ones(self.Ny-1),k=1)+np.diag(np.ones(1),k=-self.Ny+1)
        Ax=1/2*(np.diag(np.ones(self.Nx),k=0)+np.diag(np.ones(self.Nx-1),k=1)+np.diag(np.ones(1),k=-self.Nx+1))
        Ay=1/2*(np.diag(np.ones(self.Ny),k=0)+np.diag(np.ones(self.Ny-1),k=1)+np.diag(np.ones(1),k=-self.Ny+1))
        delta_x_inv=np.diag(1/self.dx,k=0)
        delta_y_inv=np.diag(1/self.dy,k=0)
        delta_x_dual=np.diag(self.dx_dual,k=0)
        delta_y_dual=np.diag(self.dy_dual,k=0)

        delta_dual=np.zeros((2*self.Nx*self.Ny,2*self.Nx*self.Ny))
        delta_dual[:self.Nx*self.Ny,:self.Nx*self.Ny]=np.kron(delta_x_dual,np.eye(self.Ny))
        delta_dual[self.Nx*self.Ny:,self.Nx*self.Ny:]=np.kron(np.eye(self.Nx),delta_y_dual)

        # Als we anisotrope media gaan bestuderen hebben we deze nodig, maar nu nog niet
        # delta_dual_inv=np.zeros((2*self.Nx*self.Ny,2*self.Nx*self.Ny))
        # delta_dual_inv[np.diag_indices_from(delta_dual_inv)]=1/delta_dual[np.diag_indices_from(delta_dual)]

        star_c_o_curl_10=np.zeros((2*self.Nx*self.Ny,self.Nx*self.Ny))
        star_c_o_curl_10[:self.Nx*self.Ny,:]=np.kron(np.eye(self.Nx),np.dot(delta_y_inv,Dy))
        star_c_o_curl_10[self.Nx*self.Ny:,:]=np.kron(-np.dot(delta_x_inv,Dx),np.eye(self.Ny))

        star_c_o_curl_01=np.zeros((self.Nx*self.Ny,2*self.Nx*self.Ny))
        star_c_o_curl_01[:,:self.Nx*self.Ny]=np.kron(-Ax,np.dot(delta_y_inv,Dy))
        star_c_o_curl_01[:,self.Nx*self.Ny:]=np.kron(np.dot(delta_x_inv,Dx),Ay)

        A_00=np.kron(Ax,Ay)

        A_11=np.zeros((2*self.Nx*self.Ny,2*self.Nx*self.Ny))
        A_11[:self.Nx*self.Ny,:self.Nx*self.Ny]=np.kron(np.eye(self.Nx),Ay)
        A_11[self.Nx*self.Ny:,self.Nx*self.Ny:]=np.kron(Ax,np.eye(self.Ny))

        eps=np.diag(self.eps,k=0)
        A_eps_dt=1/self.dt*A_00@eps # Hier hebben we normaal delta_duals nodig, maar negeren we voorlopig
        
        mu=np.zeros((2*self.Nx*self.Ny,2*self.Ny*self.Ny))
        mu[[i for i in range(self.Nx*self.Ny)],[i for i in range(self.Nx*self.Ny)]]=self.mu
        mu[[i+self.Nx*self.Ny for i in range(self.Nx*self.Ny)],[i+self.Nx*self.Ny for i in range(self.Nx*self.Ny)]]=self.mu
        A_mu_dt=1/self.dt*A_11@mu # Hier hebben we normaal delta_duals nodig, maar negeren we voorlopig
 
        self.left_matrix=np.zeros((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.left_matrix[:self.Nx*self.Ny,:self.Nx*self.Ny]=A_eps_dt
        self.left_matrix[:self.Nx*self.Ny,self.Nx*self.Ny:]=-star_c_o_curl_01/2
        self.left_matrix[self.Nx*self.Ny:,:self.Nx*self.Ny]=star_c_o_curl_10/2
        self.left_matrix[self.Nx*self.Ny:,self.Nx*self.Ny:]=A_mu_dt

        self.right_matrix=np.zeros((3*self.Nx*self.Ny,3*self.Nx*self.Ny))
        self.right_matrix[:self.Nx*self.Ny,:self.Nx*self.Ny]=A_eps_dt
        self.right_matrix[:self.Nx*self.Ny,self.Nx*self.Ny:]=star_c_o_curl_01/2
        self.right_matrix[self.Nx*self.Ny:,:self.Nx*self.Ny]=-star_c_o_curl_10/2
        self.right_matrix[self.Nx*self.Ny:,self.Nx*self.Ny:]=A_mu_dt

        self.left_matrix_inv=np.linalg.inv(self.left_matrix) # Echt superslechte methode voor oplossen stelsel, maar ik wil kijken of mijn methode werkt

    def restart(self):
        self.all_fields=np.zeros(3*self.Nx*self.Ny)
        self.n=0
    
    def update(self):
        self.source[self.xs*self.Ny+self.ys]+=self.J0*np.sin(self.Wc*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc)**2/(2*self.width**2))
        self.all_fields=self.left_matrix_inv@(self.right_matrix@self.all_fields+self.source)
        self.n+=1

    def update_loop(self,nt=None):
        if nt is None:
            nt = self.Nt
        for _ in range(nt):
            self.update()
             
    def animate(self,speed=1,repeat=False):
        Ez=np.reshape(self.all_fields[:self.Nx*self.Ny],(self.Nx,self.Ny))
        fig, ax = plt.subplots()
        im = ax.imshow(Ez.T,cmap='RdBu_r',extent=(0,np.sum(self.dx),0,np.sum(self.dy)),vmin=-0.1,vmax=0.1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0,np.sum(self.dx))
        ax.set_ylim(0,np.sum(self.dy))
        ax.set_title('Ez')
        source_marker, = ax.plot(sum(self.dx[:self.xs]), sum(self.dy[:self.ys]), 'o', color='black', label='source', markersize=2, zorder=3)
        def update(frame):
            self.update_loop(speed)
            im.set_data(np.reshape(self.all_fields[:self.Nx*self.Ny],(self.Nx,self.Ny)).T)
            return [im,source_marker]
        
        ani = FuncAnimation(fig, update, frames=self.Nt//speed, interval=int(self.dt * 1000), repeat=repeat)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, label='Ez [V/m]')
        ax.legend(loc='upper left')
        plt.show()


