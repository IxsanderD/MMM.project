import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.constants.astropyconst20 import m_e,hbar,e,h,k_B

class RTD:
    def __init__(self,dx,dt,a,b,Ly,Lz,t_max,x0,sigma_x,kx,sigma,k,N_layer,m,n,order,CFL=0.99,ABC=True):
        self.dx = dx
        self.dt = dt
        # if order == 2:
        #     self.dt = CFL*2/(2*hbar.value/(0.023*m_e.value*dx**2))
        # elif order == 4:
        #     self.dt = CFL*2/(8*hbar.value/(3*0.023*m_e.value*dx**2))
        self.CFL = CFL
        self.a = a
        self.b = b
        self.Lx = 3*a+2*b+108e-9
        self.Ly = Ly
        self.Lz = Lz
        self.t_max = t_max
        self.x0 = x0
        self.sigma_x = sigma_x
        self.kx = kx
        self.order = order
        self.m_en = m
        self.n_en = n
        self.E = hbar.value**2/(2*0.023*m_e.value)*((np.pi*n/Ly)**2+(np.pi*m/Lz)**2)
        print(f'Energy(n,m): {self.E/e.value} eV')
        self.Nx = int(self.Lx//self.dx)
        self.Nt = int(self.t_max//self.dt)
        self.C = 1/np.sqrt(np.sqrt(2*np.pi)*self.sigma_x)
        self.psi_Re = np.zeros(self.Nx)
        self.psi_Im = np.zeros(self.Nx)
        self.U = np.zeros(self.Nx)
        self.Vdc = 0
        self.xr = 0
        self.n = 0
        self.m = 0.023*m_e.value
        self.hbar = hbar.value
        # Absorbing Boundaries:
        self.U_Im = np.zeros(self.Nx)
        self.N_layer=N_layer
        self.k = k
        self.sigma = sigma
        if ABC:
            self.U_Im[:N_layer] += np.array([sigma*(i/N_layer)**k for i in range(N_layer-1,-1,-1)])
            self.U_Im[-N_layer:] += np.array([sigma*(i/N_layer)**k for i in range(N_layer)])
    
    def restart(self):
        self.psi_Re = np.zeros(self.Nx)
        self.psi_Im = np.zeros(self.Nx)
        self.psiRe_record_left = []
        self.psiRe_record_right = []
        self.psiIm_record_left = []
        self.psiIm_record_right = []
        self.n = 0
    
    def add_barriers(self,U0):
        self.U0 = U0
        self.U[int((self.a+48e-9)//self.dx):int((self.a+self.b+48e-9)//self.dx)] = U0
        self.U[int((2*self.a+self.b+48e-9)//self.dx):int((2*self.a+2*self.b+48e-9)//self.dx)] = U0
        self.Kx = np.sqrt(2*self.m*(self.E-U0)/self.hbar**2 + 0j)
        # if self.order == 2:
        #     self.dt = self.CFL*2/(2*hbar.value/(0.023*m_e.value*self.dx**2)+U0/hbar.value)
        # elif self.order == 4:
        #     self.dt = self.CFL*2/(8*hbar.value/(3*0.023*m_e.value*self.dx**2)+U0/hbar.value)
        
    def add_potential(self,V0):
        self.U[:int((self.a+48e-9)//self.dx)]  = V0*np.ones(int((self.a+48e-9)//self.dx))
        self.U[int((self.a+48e-9)//self.dx):int((2*self.a+2*self.b+48e-9)//self.dx)] += np.linspace(V0,0,int((self.a+2*self.b)//self.dx))
        self.Vdc=V0
        # if self.order == 2:
        #     self.dt = self.CFL*2/(2*hbar.value/(0.023*m_e.value*self.dx**2)+np.max(self.U)/hbar.value)
        # elif self.order == 4:
        #     self.dt = self.CFL*2/(8*hbar.value/(3*0.023*m_e.value*self.dx**2)+np.max(self.U)/hbar.value)
        
    def plot_potential(self):
        plt.plot(np.arange(self.Nx)*self.dx*1e9,self.U/e.value,label='Re')
        plt.plot(np.arange(self.Nx)*self.dx*1e9,self.U_Im/e.value,label='Im')
        plt.legend()
        plt.xlabel('x [nm]')
        plt.ylabel('U [eV]')
        plt.xlim(0,self.Lx*1e9)
        plt.show()
        
    def add_recorder(self,xr):
        self.xr = xr
        self.psiRe_record_left = []
        self.psiRe_record_right = []
        self.psiIm_record_left = []
        self.psiIm_record_right = []
    
    def show_recorder(self):
        plt.plot(np.arange(len(self.psiRe_record_left))*self.dt*1e15,np.array(self.psiRe_record_left),label='Re')
        plt.plot(np.arange(len(self.psiIm_record_left))*self.dt*1e15,np.array(self.psiIm_record_left),label='Im')
        plt.xlabel('Time [fs]')
        plt.ylabel(r'$\psi(x_r)$')
        plt.legend()
        plt.show()
        
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
        
    def update(self):
        if self.n==0:
            self.psi_Re = np.array([self.C*np.cos(self.kx*i*self.dx)*np.exp(-(i*self.dx-self.x0)**2/(4*self.sigma_x**2)) for i in range(self.Nx)])
            self.psi_Im = np.array([self.C*np.sin(self.kx*i*self.dx)*np.exp(-(i*self.dx-self.x0)**2/(4*self.sigma_x**2)) for i in range(self.Nx)])
        if self.order == 2:
            deriv2 = self.deriv2_2
        elif self.order == 4:
            deriv2 = self.deriv2_4
        self.psi_Re = (-self.hbar*self.dt/(2*self.m)*deriv2(self.psi_Im)
                              + self.dt/self.hbar*(self.U+self.E)*self.psi_Im
                              +(1-self.dt/2/self.hbar*self.U_Im)*self.psi_Re)/(1+self.dt/2/self.hbar*self.U_Im)
        self.psi_Im = (self.hbar*self.dt/(2*self.m)*deriv2(self.psi_Re)
                              - self.dt/self.hbar*(self.U+self.E)*self.psi_Re
                              +(1-self.dt/2/self.hbar*self.U_Im)*self.psi_Im)/(1+self.dt/2/self.hbar*self.U_Im)
        self.psiRe_record_left.append(self.psi_Re[int(self.xr//self.dx)])
        self.psiIm_record_left.append(self.psi_Im[int(self.xr//self.dx)])
        self.psiRe_record_right.append(self.psi_Re[int(self.xr//self.dx+1)])
        self.psiIm_record_right.append(self.psi_Im[int(self.xr//self.dx+1)])
        self.n += 1
    
    def update_loop(self):
        for _ in range(self.Nt):
            self.update()

    def spectral_source(self,max_E):
        kx=self.kx
        x0=self.x0
        sigx=self.sigma_x
        C=self.C
        x=np.linspace(0,self.Nx*self.dx,self.Nx)
        source=C*np.exp(1j*kx*x)*np.exp(-(x-x0)**2/4/sigx**2)
        kx=2*np.pi*np.fft.rfftfreq(self.Nx,d=self.dx)
        E=self.hbar**2*kx**2/2/self.m/e.value
        spec_source=np.fft.fft(source)[:len(E)]*self.dx
        mask= E<max_E
        plt.plot(E[mask],np.abs(spec_source[mask])**2)
        plt.xlabel(r'Energy [eV]')
        plt.ylabel(r'Spectrum source [$\frac{1}{m}$]')
        plt.show()
    
    def animate(self,speed=1,repeat=False):
        fig, axes = plt.subplots(2,1,figsize=(8,6),gridspec_kw={'height_ratios':[3,1]})
        ax, ax2 = axes
        im = ax.plot(np.arange(self.Nx)*self.dx*1e9,self.psi_Re**2+self.psi_Im**2)[0]
        ax.set_ylabel(r'$|\psi|^2$')
        ax.set_xlim(0,self.Lx*1e9)
        ax.set_ylim(0,self.C**2)
        ax.vlines(self.xr*1e9,0,self.C**2,color = 'red',linestyle='--',label='Recorder')

        prob_text = ax.text(0.98, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        abc_left  = self.N_layer * self.dx * 1e9
        abc_right = (self.Nx - self.N_layer) * self.dx * 1e9
        ax.axvline(abc_left,  color='gray', linestyle='--', linewidth=0.8, label='ABC start')
        ax.axvline(abc_right, color='gray', linestyle='--', linewidth=0.8)

        def update(frame):
            for _ in range(speed):
                self.update()
            im.set_data(np.arange(self.Nx)*self.dx*10**9,self.psi_Re**2+self.psi_Im**2)

            prob = np.trapezoid(self.psi_Re**2 + self.psi_Im**2, dx=self.dx)
            t_fs = self.n * self.dt * 1e15
            prob_text.set_text(f'Total prob = {prob:.4f}\nt = {t_fs:.2f} fs')

            return [im,prob_text]
        
        ani = FuncAnimation(fig, update, frames=self.Nt//speed, repeat=repeat)
        # ani.save("simulation.gif", writer="pillow", fps=10)
        ax2.plot(np.arange(self.Nx)*self.dx*1e9,self.U/e.value)
        ax2.set_xlabel('x [nm]')
        ax2.set_ylabel('U [eV]')
        ax2.set_xlim(0,self.Lx*1e9)
        plt.tight_layout()
        ax.legend()
        plt.show()
        
    def show_psi(self):
        plt.plot(np.arange(self.Nx)*self.dx*1e9,self.psi_Re,label='Re')
        plt.plot(np.arange(self.Nx)*self.dx*1e9,self.psi_Im,label='Im')
        plt.xlabel('x')
        plt.ylabel(r'$\psi$')
        plt.xlim(0,self.Lx*1e9)
        plt.legend()
        plt.show()
        
    def analytical_T(self,E_max=0.9):
        T = []
        E_array_n = np.linspace(0.01,self.U0/e.value-0.01,10000)*e.value
        kx_array_n = np.sqrt(2*self.m*E_array_n/self.hbar**2)
        Kx_array_n = np.sqrt(2*self.m*(self.U0-E_array_n)/self.hbar**2)
        for kx,Kx in zip(kx_array_n,Kx_array_n):
            M12 = 1/2*np.array([[1+1j*Kx/kx,1-1j*Kx/kx],[1-1j*Kx/kx,1+1j*Kx/kx]],dtype=complex)
            M23 = 1/2*np.array([[1-1j*kx/Kx,1+1j*kx/Kx],[1+1j*kx/Kx,1-1j*kx/Kx]],dtype=complex)
            M1 = np.array([[np.exp(Kx*self.b),0],[0,np.exp(-Kx*self.b)]],dtype=complex)
            M2 = np.array([[np.exp(-1j*kx*self.a),0],[0,np.exp(1j*kx*self.a)]],dtype=complex)
            M = M12@M1@M23@M2@M12@M1@M23
            T.append(1/np.abs(M[0,0])**2)
        E_array_p = np.linspace(self.U0/e.value+0.01,E_max,10000)*e.value
        kx_array_p = np.sqrt(2*self.m*E_array_p/self.hbar**2)
        Kx_array_p = np.sqrt(2*self.m*(E_array_p-self.U0)/self.hbar**2)
        for kx,Kx in zip(kx_array_p,Kx_array_p):
            M12 = 1/2*np.array([[1+Kx/kx,1-Kx/kx],[1-Kx/kx,1+Kx/kx]],dtype=complex)
            M23 = 1/2*np.array([[1+kx/Kx,1-kx/Kx],[1-kx/Kx,1+kx/Kx]],dtype=complex)
            M1 = np.array([[np.exp(-1j*Kx*self.b),0],[0,np.exp(1j*Kx*self.b)]],dtype=complex)
            M2 = np.array([[np.exp(-1j*kx*self.a),0],[0,np.exp(1j*kx*self.a)]],dtype=complex)
            M = M12@M1@M23@M2@M12@M1@M23
            T.append(1/np.abs(M[0,0])**2)
        E_array=np.concatenate((E_array_n,E_array_p))
        return E_array,np.array(T)
    
    def psi_freq(self, psi_Re, psi_Im, eta=None):
        N = len(psi_Re)
        E = np.fft.fftfreq(N, d=self.dt)
        psi_Re_freq=np.real(np.fft.fft(np.array(psi_Re)))
        psi_Im_freq=np.real(np.fft.fft(np.array(psi_Im)))
        return np.concatenate((E[:(len(E)+1)//2],E[(len(E)+1)//2:]+1/self.dt))*2*np.pi*self.hbar-self.E, psi_Re_freq, psi_Im_freq
    
    def J_time(self):
        N = 1e26/(self.Ly*self.Lz)
        Re_left = np.array(self.psiRe_record_left)
        Im_left = np.array(self.psiIm_record_left)
        Re_right = np.array(self.psiRe_record_right)
        Im_right = np.array(self.psiIm_record_right)
        J = Re_left*Im_right - Im_left*Re_right
        t = np.arange(len(J))*self.dt
        return t, N*e.value*self.hbar/(self.m*self.dx)*np.array(J)
    
    def J_freq(self): # To be continued
        N = 1e26/(self.Ly*self.Lz)
        E, psi_Re_freq, psi_Im_freq = self.psi_freq(self.psiRe_record_left,self.psiIm_record_left)
        diff_psi_Re = (np.array(self.psiRe_record_right)-np.array(self.psiRe_record_left))/self.dx
        diff_psi_Im = (np.array(self.psiIm_record_right)-np.array(self.psiIm_record_left))/self.dx
        _, diff_psi_Re_freq, diff_psi_Im_freq = self.psi_freq(diff_psi_Re,diff_psi_Im)
        return E, self.hbar/self.m*(psi_Re_freq*diff_psi_Im_freq-psi_Im_freq*diff_psi_Re_freq)
    
    def Transmission(self):
        E_num, J_bar = self.J_freq()

        free = RTD(self.dx,self.dt,self.a,self.b,self.Ly,self.Lz,self.t_max,self.x0,self.sigma_x,self.kx,self.sigma,self.k,self.N_layer,self.m_en,self.n_en,order=self.order,ABC=True)
        free.add_potential(self.Vdc)
        free.add_recorder(self.xr)

        free.update_loop()
        # free.show_recorder()
        E_num, J_free = free.J_freq()
        mask=E_num/e.value<0.9

        T_num = np.abs(J_bar[mask]/J_free[mask])
        return E_num[mask]/e.value,T_num
    
    def IV(self,E,T,mu_l=22.436e-3*e.value,Te=0):
        El = mu_l - e.value*self.Vdc - 6*k_B.value*Te
        Er = mu_l + 6*k_B.value*Te
        mask = (El<E)&(Er>E)
        if Te==0:
            I = 2*e.value/h.value*np.trapezoid(T[mask],E[mask],dx=E[1]-E[0])
            return I
        else:
            f_L = 1/(np.exp((E-mu_l)/k_B.value/Te)+1)
            f_R = 1/(np.exp((E-mu_l+e.value*self.Vdc)/k_B.value/Te)+1)
            I = 2*e.value/h.value*np.trapezoid(T[mask]*(f_L[mask]-f_R[mask]),E[mask],dx=E[1]-E[0])
            return I