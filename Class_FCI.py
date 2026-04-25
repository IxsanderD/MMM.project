import numpy as np
from scipy.sparse import csr_array, diags, eye, kron, bmat
from scipy.sparse.linalg import splu, inv
from scipy.constants import epsilon_0, mu_0, c
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.special import hankel2
import time

class FCI:
    def __init__(self,Nt,dx,dy,dt,k_max,sigma_max,drude=False):
        self.Nx=len(dx)
        self.Ny=len(dy)
        self.Nt=Nt
        self.dx=dx
        self.dy=dy
        self.dx_dual = np.append((self.dx[1:]+self.dx[:-1])/2,(self.dx[0]+self.dx[-1])/2)
        self.dy_dual = np.append((self.dy[1:]+self.dy[:-1])/2,(self.dy[0]+self.dy[-1])/2)
        self.dt=dt
        self.drude=drude
        if drude:
            self.all_fields=np.zeros(8*self.Nx*self.Ny)
        else:
            self.all_fields=np.zeros(6*self.Nx*self.Ny) 
        self.source_index=[]
        self.J0 = []
        self.tc = []
        self.width = []
        self.Wc = []
        self.eps=np.ones(self.Nx*self.Ny)*epsilon_0
        self.mu=np.ones(self.Nx*self.Ny)*mu_0
        self.c = c
        self.sigma=np.zeros(self.Nx*self.Ny)
        self.gamma=np.zeros(self.Nx*self.Ny)
        self.n=0
        self.kmax=k_max
        self.sigmamax=sigma_max
        self.materials = []

    def restart(self):
        self.all_fields=np.zeros(6*self.Nx*self.Ny)
        self.n=0
        self.recorded_Ez = []

    def add_source(self,xs,ys,J0,tc,w,Wc=None):
        self.source_index.append((xs,ys))
        self.J0.append(J0)
        self.tc.append(tc)
        self.width.append(w)
        self.Wc.append(Wc)
        time = np.arange(self.Nt)*self.dt
        if Wc==None:
            self.applied_source = J0*np.exp(-(time-tc)**2/(2*w**2))
        else:
            self.applied_source = J0*np.sin(Wc*time)*np.exp(-(time-tc)**2/(2*w**2))

    def add_recorder(self,xr,yr):
        self.xr = xr
        self.yr = self.Ny - yr - 1
        self.recorded_Ez = []

    def add_material(self,x_s,x_e,y_s,y_e,eps_r,mu_r,sigma,gamma=0):
        eps=np.reshape(self.eps,(self.Nx,self.Ny))
        eps[x_s:x_e,y_s:y_e]=eps_r*self.eps[0]
        self.eps=np.ravel(eps)

        mu=np.reshape(self.mu,(self.Nx,self.Ny))
        mu[x_s:x_e,y_s:y_e]=mu_r*self.mu[0]
        self.mu=np.ravel(mu)

        sig=np.reshape(self.sigma,(self.Nx,self.Ny))
        sig[x_s:x_e,y_s:y_e]=sigma
        self.sigma=np.ravel(sig)

        gam=np.reshape(self.gamma,(self.Nx,self.Ny))
        gam[x_s:x_e,y_s:y_e]=gamma
        self.gamma=np.ravel(gam)

        self.materials.append((x_s, x_e, y_s, y_e))
    
    def construct_update_matrix(self):
        Dx=-diags(np.ones(self.Nx),offsets=0,format='csr')+diags(np.ones(self.Nx-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Nx+1,format='csr')
        Dy=-diags(np.ones(self.Ny),offsets=0,format='csr')+diags(np.ones(self.Ny-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Ny+1,format='csr')
        Ax=1/2*(diags(np.ones(self.Nx),offsets=0,format='csr')+diags(np.ones(self.Nx-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Nx+1,format='csr'))
        Ay=1/2*(diags(np.ones(self.Ny),offsets=0,format='csr')+diags(np.ones(self.Ny-1),offsets=1,format='csr')+diags(np.ones(1),offsets=-self.Ny+1,format='csr'))
        delta_x_inv=diags(1/self.dx,offsets=0,format='csr')
        delta_y_inv=diags(1/self.dy,offsets=0,format='csr')
        k_array=np.array([1+(self.kmax-1)*(i/10)**4 for i in range(1,11)])
        K_y=np.ones((self.Nx,self.Ny))
        K_y[:,self.Ny-10:]=np.tile(k_array,(self.Nx,1))
        K_y[:,:10]=np.tile(np.flip(k_array),(self.Nx,1))
        K_x=np.ones((self.Nx,self.Ny))
        k_array_v=np.reshape(k_array,(10,1))
        K_x[self.Nx-10:,:]=np.repeat(k_array_v,self.Ny,axis=1)
        k_array_v=np.reshape(np.flip(k_array),(10,1))
        K_x[:10,:]=np.repeat(k_array_v,self.Ny,axis=1)
        sigma_array=np.array([self.sigmamax*(i/10)**4 for i in range(1,11)])
        sigma_y=np.zeros((self.Nx,self.Ny))
        sigma_y[:,self.Ny-10:]=np.tile(sigma_array,(self.Nx,1))
        sigma_y[:,:10]=np.tile(np.flip(sigma_array),(self.Nx,1))
        sigma_array_v=np.reshape(sigma_array,(10,1))
        sigma_x=np.zeros((self.Nx,self.Ny))
        sigma_x[self.Nx-10:,:]=np.repeat(sigma_array_v,self.Ny,axis=1)
        sigma_array_v=np.reshape(np.flip(sigma_array),(10,1))
        sigma_x[:10,:]=np.repeat(sigma_array_v,self.Ny,axis=1)

        K_x=np.ravel(K_x)
        K_y=np.ravel(K_y)
        sigma_x=np.ravel(sigma_x)
        sigma_y=np.ravel(sigma_y)
        b_x_inv=1/(K_x/self.dt+sigma_x/2/self.eps)
        b_y_inv=1/(K_y/self.dt+sigma_y/2/self.eps)

        K_y_dt=diags(K_y,offsets=0,format='csr')/self.dt
        K_x_dt=diags(K_x,offsets=0,format='csr')/self.dt
        sigma_x_2=diags(sigma_x,offsets=0,format='csr')/2
        sigma_y_2=diags(sigma_y,offsets=0,format='csr')/2
        eps=diags(self.eps,offsets=0,format='csr')
        eps_inv=diags(1/self.eps,offsets=0,format='csr')
        mu=diags(self.mu,offsets=0,format='csr')
        sigma=diags(self.sigma,offsets=0,format='csr')

        A_x=kron(np.eye(self.Nx),Ay,format='csr')
        A_y=kron(Ax,np.eye(self.Ny),format='csr')
        A_z=kron(Ax,Ay,format='csr')

        # block (3,0)
        A30=-(K_y_dt+sigma_y_2@eps_inv)
        B30=-(K_y_dt-sigma_y_2@eps_inv)

        # block (3,3)
        # A33=eye(self.Nx*self.Ny,format='csr')/self.dt
        B33=eye(self.Nx*self.Ny,format='csr')/self.dt

        # block (2,3)
        A23=A_z@(eps@K_x_dt+sigma_x_2+sigma/2)
        B23=A_z@(eps@K_x_dt-sigma_x_2-sigma/2)

        # block (2,1)
        A21=kron(Ax,delta_y_inv@Dy,format='csr')/2
        B21=-kron(Ax,delta_y_inv@Dy,format='csr')/2

        # block (2,2)
        A22=-kron(delta_x_inv@Dx,Ay,format='csr')/2
        B22=kron(delta_x_inv@Dx,Ay,format='csr')/2

        # block (4,1)
        A41=-eye(self.Nx*self.Ny,format='csr')/self.dt
        B41=-eye(self.Nx*self.Ny,format='csr')/self.dt

        # block (4,4)
        # A44=K_x_dt+sigma_x_2@eps_inv
        B44=K_x_dt-sigma_x_2@eps_inv

        # block (0,0)
        A00=kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2
        B00=-kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2

        # block (0,4)
        A04=A_x@(mu@K_y_dt+mu@sigma_y_2@eps_inv)
        B04=A_x@(mu@K_y_dt-mu@sigma_y_2@eps_inv)

        # block (5,2)
        A52=-(K_x_dt+sigma_x_2@eps_inv)
        B52=-(K_x_dt-sigma_x_2@eps_inv)

        # block (5,5)
        # A55=(K_y_dt+sigma_y_2@eps_inv)
        B55=(K_y_dt-sigma_y_2@eps_inv)

        # block (1,0)
        A10=-kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2
        B10=kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2

        # block (1,5)
        A15=A_y@mu/self.dt
        B15=A_y@mu/self.dt

        Z=csr_array((self.Nx*self.Ny, self.Nx*self.Ny))

        self.M11=bmat([[A00,Z,Z],[A10,Z,Z],[Z,A21,A22]],format='csr')
        self.M12=bmat([[Z,A04,Z],[Z,Z,A15],[A23,Z,Z]],format='csr')
        self.M21=bmat([[A30,Z,Z],[Z,A41,Z],[Z,Z,A52]],format='csr')
        self.right_matrix=bmat([[B00,Z,Z,Z,B04,Z],[B10,Z,Z,Z,Z,B15],[Z,B21,B22,B23,Z,Z],[B30,Z,Z,B33,Z,Z],[Z,B41,Z,Z,B44,Z],[Z,Z,B52,Z,Z,B55]],format='csr')
        
        A33_inv=eye(self.Nx*self.Ny,format='csr')*self.dt
        A44_inv=diags(b_x_inv,offsets=0,format='csr')
        A55_inv=diags(b_y_inv,offsets=0,format='csr')
        self.M22_inv=bmat([[A33_inv,Z,Z],[Z,A44_inv,Z],[Z,Z,A55_inv]],format='csc')
        S=bmat([[A00,-A04@A44_inv@A41,Z],[A10,Z,-A15@A55_inv@A52],[-A23@A33_inv@A30,A21,A22]],format='csc')
        self.S_LU=splu(S)

    def construct_update_matrix_drude(self):
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

        K_x=np.ravel(K_x)
        K_y=np.ravel(K_y)
        sigma_x=np.ravel(sigma_x)
        sigma_y=np.ravel(sigma_y)
        b_x_inv=1/(K_x/self.dt+sigma_x/2/self.eps)
        b_y_inv=1/(K_y/self.dt+sigma_y/2/self.eps)

        K_y_dt=diags(K_y,offsets=0,format='csr')/self.dt
        K_x_dt=diags(K_x,offsets=0,format='csr')/self.dt
        sigma_x_2=diags(sigma_x,offsets=0,format='csr')/2
        sigma_y_2=diags(sigma_y,offsets=0,format='csr')/2
        eps=diags(self.eps,offsets=0,format='csr')
        eps_inv=diags(1/self.eps,offsets=0,format='csr')
        mu=diags(self.mu,offsets=0,format='csr')
        sigma=diags(self.sigma,offsets=0,format='csr')
        gamma=diags(self.gamma,offsets=0,format='csr')

        A_x=kron(np.eye(self.Nx),Ay,format='csr')
        A_y=kron(Ax,np.eye(self.Ny),format='csr')
        A_z=kron(Ax,Ay,format='csr')

        # block (7,7)
        # A77=-(K_y_dt+sigma_y_2@eps_inv)
        B77=-(K_y_dt-sigma_y_2@eps_inv)

        # block (7,4)
        A74=eye(self.Nx*self.Ny,format='csr')/self.dt
        B74=eye(self.Nx*self.Ny,format='csr')/self.dt

        # block (1,5)
        A15=kron(Ax,delta_y_inv@Dy,format='csr')/2
        B15=-kron(Ax,delta_y_inv@Dy,format='csr')/2

        # block (1,6)
        A16=-kron(delta_x_inv@Dx,Ay,format='csr')/2
        B16=kron(delta_x_inv@Dx,Ay,format='csr')/2

        # block (1,1)
        A11=A_z@eps/self.dt
        B11=A_z@eps/self.dt

        # block (1,0)
        A10=A_z/2
        B10=-A_z/2

        # block (5,5)
        # A55=-eye(self.Nx*self.Ny,format='csr')/self.dt
        B55=-eye(self.Nx*self.Ny,format='csr')/self.dt

        # block (5,2)
        A52=(K_x_dt+sigma_x_2@eps_inv)
        B52=(K_x_dt-sigma_x_2@eps_inv)

        # block (2,7)
        A27=kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2
        B27=-kron(eye(self.Nx,format='csr'),delta_y_inv@Dy,format='csr')/2

        # block (2,2)
        A22=A_x@(mu@K_y_dt+mu@sigma_y_2@eps_inv)
        B22=A_x@(mu@K_y_dt-mu@sigma_y_2@eps_inv)

        # block (6,6)
        # A66=-(K_x_dt+sigma_x_2@eps_inv)
        B66=-(K_x_dt-sigma_x_2@eps_inv)

        # block (6,3)
        A63=(K_y_dt+sigma_y_2@eps_inv)
        B63=(K_y_dt-sigma_y_2@eps_inv)

        # block (3,7)
        A37=-kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2
        B37=kron(delta_x_inv@Dx,eye(self.Ny,format='csr'),format='csr')/2

        # block (3,3)
        A33=A_y@mu/self.dt
        B33=A_y@mu/self.dt

        # block (4,4)
        # A44=-(K_x_dt+sigma_x_2@eps_inv)
        B44=-(K_x_dt-sigma_x_2@eps_inv)

        # block (4,1)
        A41=eye(self.Nx*self.Ny,format='csr')/self.dt
        B41=eye(self.Nx*self.Ny,format='csr')/self.dt

        # Block (0,1)
        A01=sigma/2
        B01=-sigma/2

        # Block (0,0)
        A00=-(gamma/self.dt)-eye(self.Nx*self.Ny,format='csr')/2
        B00=-(gamma/self.dt)+eye(self.Nx*self.Ny,format='csr')/2

        Z=csr_array((self.Nx*self.Ny, self.Nx*self.Ny))

        self.M11=bmat([[A00,A01,Z,Z],[A10,A11,Z,Z],[Z,Z,A22,Z],[Z,Z,Z,A33]],format='csr')
        self.M12=bmat([[Z,Z,Z,Z],[Z,A15,A16,Z],[Z,Z,Z,A27],[Z,Z,Z,A37]],format='csr')
        self.M21=bmat([[Z,A41,Z,Z],[Z,Z,A52,Z],[Z,Z,Z,A63],[Z,Z,Z,Z]],format='csr')
        self.right_matrix=bmat([[B00,B01,Z,Z,Z,Z,Z,Z],[B10,B11,Z,Z,Z,B15,B16,Z],[Z,Z,B22,Z,Z,Z,Z,B27],[Z,Z,Z,B33,Z,Z,Z,B37],[Z,B41,Z,Z,B44,Z,Z,Z],[Z,Z,B52,Z,Z,B55,Z,Z],[Z,Z,Z,B63,Z,Z,B66,Z],[Z,Z,Z,Z,B74,Z,Z,B77]],format='csr')
        self.A00_inv=-diags(1/(self.gamma/self.dt+1/2),offsets=0,format='csr')
        A44_inv=-diags(b_x_inv,offsets=0,format='csr')
        A55_inv=-eye(self.Nx*self.Ny,format='csr')*self.dt
        A66_inv=-diags(b_x_inv,offsets=0,format='csr')
        A77_inv=-diags(b_y_inv,offsets=0,format='csr')
        self.M22_inv=bmat([[A44_inv,Z,Z,Z],[Z,A55_inv,Z,Z],[Z,Z,A66_inv,Z],[-A77_inv@A74@A44_inv,Z,Z,A77_inv]])
        self.S12=bmat([[A01,Z,Z]],format='csr')
        self.S21=bmat([[A10],[Z],[Z]],format='csr')
        Ss=bmat([[A11-A10@self.A00_inv@A01,-A15@A55_inv@A52,-A16@A66_inv@A52],[A27@A77_inv@A74@A44_inv@A41,A22,Z],[A37@A77_inv@A74@A44_inv@A41,Z,A33]],format='csc')
        self.Ss_LU=splu(Ss)

    def construct_matrices(self):
        if self.drude:
            self.construct_update_matrix_drude()
        else:
            self.construct_update_matrix()
    
    def update(self):
        b1=self.right_matrix[:3*self.Nx*self.Ny,:]@self.all_fields
        for i in range(len(self.source_index)):
            if self.Wc[i]==None:
                b1[2*self.Nx*self.Ny+self.source_index[i][0]*self.Ny+self.source_index[i][1]]-=self.J0[i]*np.exp(-(self.n*self.dt-self.tc[i])**2/(2*self.width[i]**2))
            else:
                b1[2*self.Nx*self.Ny+self.source_index[i][0]*self.Ny+self.source_index[i][1]]-=self.J0[i]*np.sin(self.Wc[i]*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc[i])**2/(2*self.width[i]**2))
        b2=self.right_matrix[3*self.Nx*self.Ny:,:]@self.all_fields
        self.all_fields[:3*self.Nx*self.Ny]=self.S_LU.solve(b1-self.M12@self.M22_inv@b2)
        self.all_fields[3*self.Nx*self.Ny:]=self.M22_inv@(b2-self.M21@self.all_fields[:3*self.Nx*self.Ny])
        self.n+=1
        Ez=np.reshape(self.all_fields[:self.Nx*self.Ny],(self.Nx,self.Ny))
        self.recorded_Ez.append(Ez[self.xr,self.yr])
    
    def update_drude(self):
        b1=self.right_matrix[:4*self.Nx*self.Ny,:]@self.all_fields
        for i in range(len(self.source_index)):
            if self.Wc[i]==None:
                b1[self.Nx*self.Ny+self.source_index[i][0]*self.Ny+self.source_index[i][1]]-=self.J0[i]*np.exp(-(self.n*self.dt-self.tc[i])**2/(2*self.width[i]**2))
            else:
                b1[self.Nx*self.Ny+self.source_index[i][0]*self.Ny+self.source_index[i][1]]-=self.J0[i]*np.sin(self.Wc[i]*self.n*self.dt)*np.exp(-(self.n*self.dt-self.tc[i])**2/(2*self.width[i]**2))
        b2=self.right_matrix[4*self.Nx*self.Ny:,:]@self.all_fields
        p=b1-self.M12@self.M22_inv@b2
        p1=p[:self.Nx*self.Ny]
        p2=p[self.Nx*self.Ny:]
        self.all_fields[self.Nx*self.Ny:4*self.Nx*self.Ny]=self.Ss_LU.solve(p2-self.S21@self.A00_inv@p1)
        self.all_fields[:self.Nx*self.Ny]=self.A00_inv@(p1-self.S12@self.all_fields[self.Nx*self.Ny:4*self.Nx*self.Ny])
        self.all_fields[4*self.Nx*self.Ny:]=self.M22_inv@(b2-self.M21@self.all_fields[:4*self.Nx*self.Ny])
        self.n+=1
        Ez=np.reshape(self.all_fields[7*self.Nx*self.Ny:],(self.Nx,self.Ny))
        self.recorded_Ez.append(Ez[self.xr,self.yr])

    def update_loop(self,nt=None):
        if nt is None:
            nt = self.Nt
        for _ in range(nt):
            self.update()
    
    def update_loop_drude(self,nt=None):
        if nt is None:
            nt = self.Nt
        for _ in range(nt):
            self.update_drude()

    def animate(self, speed=1, repeat=False):
        Ez = np.reshape(self.all_fields[:self.Nx*self.Ny], (self.Nx, self.Ny))
        
        x_edges = np.concatenate([[0], np.cumsum(self.dx)])
        y_edges = np.concatenate([[0], np.cumsum(self.dy)])
        
        fig, ax = plt.subplots()
        im = ax.pcolormesh(x_edges, y_edges, Ez.T, cmap='RdBu_r', vmin=-0.05, vmax=0.05)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.set_title('Ez')

        for mat in self.materials:
            x_start, x_end, y_start, y_end = mat
            x0 = np.sum(self.dx[:x_start])
            x1 = np.sum(self.dx[:x_end])
            y0 = np.sum(self.dy[:y_start])
            y1 = np.sum(self.dy[:y_end])
            width = x1 - x0
            height = y1 - y0
            rect = plt.Rectangle((x0, y0), width, height,  linewidth=1, edgecolor='black', facecolor='none', zorder=2)
            ax.add_patch(rect)

        for i in range(len(self.source_index)):
            source_marker, = ax.plot(sum(self.dx[:self.source_index[i][0]]), sum(self.dy[:self.source_index[i][1]]), 'o', color='black', label='source', markersize=2, zorder=3)
            rec1, = ax.plot(sum(self.dx[:self.xr]), sum(self.dy[:self.yr]), 'x', color='red', label='recorder 1', zorder=3, markersize=6)

        def update(frame):
            if self.drude:
                self.update_loop_drude(speed)
                Ez = np.reshape(self.all_fields[7*self.Nx*self.Ny:], (self.Nx, self.Ny))
            else:
                self.update_loop(speed)
                Ez = np.reshape(self.all_fields[:self.Nx*self.Ny], (self.Nx, self.Ny))
            im.set_array(Ez.T.ravel())
            ax.set_title('Ez at t = {:.2f} µs'.format(self.n*self.dt*1e6))
            return [im, source_marker, rec1]

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

    def analytical_sol(self, p_all = True, f_lim = None):

        if p_all:
            plt.plot(self.recorded_Ez, label='Electric field at recorder (V/m)')
            plt.plot(self.applied_source, label='Applied source ($A/m^2$)')
            plt.xlabel('Time (s)')
            plt.legend()
            plt.title('Time domain response')
            plt.show()

        E_freq_sim = np.fft.rfft(self.recorded_Ez)*self.dt
        source_freq = np.fft.rfft(self.applied_source)*self.dt
        omega = 2*np.pi*np.fft.rfftfreq(len(self.recorded_Ez), self.dt)

        delta_x = np.sum(self.dx[:self.source_index[0][0]]) - np.sum(self.dx[:self.xr])
        delta_y = np.sum(self.dy[:self.source_index[0][1]]) - np.sum(self.dy[:self.yr])
        print(delta_x, delta_y)

        E_freq_ana = -self.J0[0]*omega*mu_0/4*hankel2(0, omega/self.c*np.sqrt(delta_x**2+delta_y**2))
        E_freq_ana[0] = 0

        E_max = np.max(np.abs(source_freq))
        if f_lim is None:
            mask = (np.abs(source_freq) > 0.005*E_max)
        else:
            mask = (omega <= f_lim)

        if p_all:
            plt.plot(omega, np.abs(E_freq_sim), label='Electric field at recorder (V/m)')
            plt.plot(omega, np.abs(source_freq), label='Applied source current density ($A/m^2$)')
            plt.xlabel('Frequency (rad/s)')
            plt.legend()
            plt.title('Frequency domain response')
            plt.show()

        plt.plot(omega[mask], np.abs(E_freq_sim[mask]/source_freq[mask]*self.Nx*self.Ny/np.sum(self.dx)/np.sum(self.dy)), label='Numerical response')
        plt.plot(omega[mask], np.abs(E_freq_ana/self.J0)[mask], label='Analytical response')
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('$|E_z/J|$')
        plt.legend()
        plt.title('Frequency response comparison')
        plt.show()