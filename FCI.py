import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c
from Class_FCI import FCI
import time

# start=time.perf_counter()
L = 100
Nx=301
Ny=301
Nt=300
dx=np.ones(Nx)*L/Nx
dy=np.ones(Ny)*L/Ny
c0=1
J0 = 1 # A/m^2
width = np.sum(dx)/(10*c0)
tc = 5*width
tf= 5*tc
dt= tf/Nt
Wc = 5/width
Wc = 2*np.pi*c0/(7*dx[0])
width = 5/Wc
tc = 5*width
dt = 5*tc/Nt

k_max=1
sigma_max=1 #(m+1)/(150*np.pi*dx[0])

xs = Nx//2
ys = Ny//2


start=time.perf_counter()
solver=FCI(Nt,dx,dy,dt,k_max,sigma_max,drude=True)
# solver.add_material(3*Nx//5,4*Nx//5,3*Ny//5,4*Ny//5,1,1,1000)
solver.construct_matrices()
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xs+Nx//4,ys)
# solver.update_loop()
# solver.show_recorder()
stop=time.perf_counter()
print(stop-start)
solver.animate()

###
# Analytical verification
###

# xs = Nx//4
# ys = 3*Ny//4
# xr = Nx//4
# yr = Ny//4

# solver=FCI(Nt,dx,dy,dt,k_max,sigma_max)
# solver.construct_update_matrix()
# end=time.perf_counter()
# print(f"Runtime: {end - start:.6f} seconds")
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xr,yr)

# solver.update_loop()
# solver.analytical_sol(plot_all=True, frequency_limit = None)

###
# Shielding by copper
###

# L=30*10**(-3)
# Nx=61
# Ny=61
# Nt=500
# dx=np.ones(Nx)*L/Nx
# dy=np.ones(Ny)*L/Ny
# dx[Nx//2-5:Nx//2+5]=dx[Nx//2-5:Nx//2+5]/1000
# c0=c
# J0 = 5 # A/m^2
# width = np.sum(dx)/(10*c0)
# tc = 5*width
# tf= 30*tc
# dt= tf/Nt
# Wc = 5/width
# sigma_c=5.96*10**7
# gamma=25*10**(-15)

# m=4
# k_max=1
# sigma_max=20 #(m+1)/(150*np.pi*dx[0])

# xs = Nx//4
# ys = Ny//2

# xr = Nx//2+15
# yr = Ny//2

# solver=FCI(Nt,dx,dy,dt,k_max,sigma_max,drude=False)
# solver.add_source(xs,ys,J0,tc,width)
# solver.add_recorder(xr,yr)
# solver.construct_matrices()
# solver.update_loop()
# # solver.animate()

# Ez_unshielded=np.fft.rfft(solver.recorded_Ez)

# solver.restart()
# solver.add_source(xs,ys,J0,tc,width)
# solver.add_recorder(xr,yr)
# solver.add_material(Nx//2-5,Nx//2+5,10,Ny-10,1,1,sigma_c)
# solver.construct_matrices()
# solver.update_loop()
# # solver.animate()

# Ez_shielded=np.fft.rfft(solver.recorded_Ez)

# omega=2*np.pi*np.fft.rfftfreq(Nt,dt)
# mask=np.where(omega<3/width)
# SE=20*np.log10(np.abs(Ez_unshielded[mask]/Ez_shielded[mask]))
# plt.plot(omega[mask]*10**(-9)/2/np.pi,SE[mask])
# plt.title(r'SE of copper')
# plt.xlabel(r'Angular frequency $\omega$ [GHz]')
# plt.ylabel(r'Shielding efficieny SE [dB]')
# plt.grid()
# plt.show()