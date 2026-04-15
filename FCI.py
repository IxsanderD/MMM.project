import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0
from Class_FCI import FCI
import time
from scipy.sparse.linalg import lsqr
from scipy.special import hankel2

start=time.perf_counter()

L = 1
Nx=51
Ny=51
Nt=500
dx=np.ones(Nx)*L/Nx
dy=np.ones(Ny)*L/Ny
c0=1
eps=1
mu=1
J0 = 10
# width = np.sum(dx)/(10*c0)
# tc = 5*width
# tf= 5*tc
# dt=tf/Nt
# Wc = 5/width
Wc = 2*np.pi*c0/(7*dx[0])
width = 5/Wc
tc = 5*width
dt = 5*tc/Nt

m = 4
k_max=1
sigma_max= 100 #(m+1)/(150*np.pi*dx[0])

# xs = Nx//2
# ys = Ny//2

# solver=FCI(Nx,Ny,Nt,dx,dy,dt,eps,mu,k_max,sigma_max)
# solver.add_material(3*Nx//5,4*Nx//5,3*Ny//5,4*Ny//5,2,2,1)
# solver.construct_update_matrix()
# end=time.perf_counter()
# print(f"Runtime: {end - start:.6f} seconds")
# solver.add_source(xs,ys,J0,tc,width,Wc)
# solver.add_recorder(xs,ys)
# # solver.update_loop()
# # solver.show_recorder()
# solver.animate()

# import plotly.express as px

# Example matrix (replace with your own)
# matrix = solver.left_matrix.toarray()

# fig = px.imshow(
#     matrix,
#     color_continuous_scale='viridis',
#     aspect='auto'
# )

# fig.update_traces(
#     hovertemplate="Row: %{y}<br>Col: %{x}<br>Value: %{z}<extra></extra>"
# )

# fig.show()

###
# Analytical verification
###

xs = Nx//4
ys = 3*Ny//4
xr = Nx//4
yr = Ny//4

solver=FCI(Nx,Ny,Nt,dx,dy,dt,eps,mu,k_max,sigma_max)
solver.construct_update_matrix()
end=time.perf_counter()
print(f"Runtime: {end - start:.6f} seconds")
solver.add_source(xs,ys,J0,tc,width,Wc)
solver.add_recorder(xr,yr)

# # Plot source in time and frequency domain
# plt.plot(np.arange(solver.Nt)*solver.dt, solver.applied_source, label='Applied source')
# plt.xlabel('Time (s)')
# plt.ylabel('Applied source (A/m^2)')
# plt.title('Time domain applied source')
# plt.legend()
# plt.show()
# plt.plot(2*np.pi*np.fft.rfftfreq(solver.Nt, solver.dt), np.abs(np.fft.rfft(solver.applied_source)*dt), label='Applied source (frequency domain)')
# plt.xlabel('Frequency (rad/s)')
# plt.ylabel('Applied source (A/m^2)')
# plt.title('Frequency domain applied source')
# plt.legend()
# plt.show()
# solver.animate()

solver.restart()
solver.update_loop()
# solver.show_recorder()

# Plot time domain response
plt.plot(solver.recorded_Ez, label='Simulated recorder')
plt.plot(solver.applied_source, label='Applied source')
plt.xlabel('Time (s)')
plt.ylabel('|E_z|')
plt.legend()
plt.title('Time domain response')
plt.show()

E_freq_sim = np.fft.rfft(solver.recorded_Ez)*dt
source_freq = np.fft.rfft(solver.applied_source)*dt
omega = 2*np.pi*np.fft.rfftfreq(len(solver.recorded_Ez), dt)

# Source and recorder distance
delta_x = np.sum(solver.dx[:xs]) - np.sum(solver.dx[:xr])
delta_y = np.sum(solver.dy[:ys]) - np.sum(solver.dy[:yr])
print(delta_x, delta_y)

# Analytical solution
E_freq_ana = -J0*omega/4*hankel2(0, omega/c0*np.sqrt(delta_x**2+delta_y**2))
E_freq_ana[0] = 0

# Restrict to bandwidth of the source
E_max = np.max(np.abs(source_freq))
mask = (np.abs(source_freq) > 0.005*E_max)
# mask = omega < 250
# mask = np.ones(len(source_freq), dtype=bool)

# Rescale
E_freq_sim *= np.mean(np.abs(E_freq_ana[mask]/J0/E_freq_sim[mask]*source_freq[mask]))

# Plot frequency domain response
plt.plot(omega, np.abs(E_freq_sim), label='Electric field at recorder (V/m)')
plt.plot(omega, np.abs(source_freq), label='Applied source current density (A/m^2)')
plt.xlabel('Frequency (rad/s)')
plt.legend()
plt.title('Frequency domain response')
plt.show()

# Compare with analytical solution
plt.plot(omega[mask], np.abs(E_freq_sim/source_freq)[mask], label='Numerical response (rescaled)')
plt.plot(omega[mask], np.abs(E_freq_ana/J0)[mask], label='Analytical response')
plt.xlabel('Frequency (rad/s)')
plt.ylabel('|E_z/J|')
plt.legend()
plt.title('Frequency response comparison')
plt.show()