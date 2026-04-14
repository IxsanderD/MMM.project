import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0
from Class_FCI import FCI
import time
from scipy.special import hankel2

start=time.perf_counter()

Nx=70
Ny=70
Nt=100
dx=np.ones(Nx)
dy=np.ones(Ny)
c0=1
eps=1
mu=1
J0 = 10
width = np.sum(dx)/(10*c0)
tc = 5*width
tf= 5*tc
dt=tf/Nt
k_max=1
sigma_max=1

# xs = Nx//2
# ys = Ny//2

# solver=FCI(Nx,Ny,Nt,dx,dy,dt,eps,mu,k_max,sigma_max)
# solver.add_material(3*Nx//5,4*Nx//5,3*Ny//5,4*Ny//5,2,2,1)
# solver.construct_update_matrix()
# end=time.perf_counter()
# print(f"Runtime: {end - start:.6f} seconds")
# solver.add_source(xs,ys,J0,tc,width)
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
solver.add_source(xs,ys,J0,tc,width)
solver.add_recorder(xr,yr)
solver.animate()

solver.restart()
solver.update_loop()
# solver.show_recorder()

# # Plot time domain response
# plt.plot(solver.recorded_Ez, label='Simulated recorder')
# plt.plot(solver.applied_source, label='Applied source')
# plt.xlabel('Time (s)')
# plt.ylabel('|E_z|')
# plt.legend()
# plt.title('Time domain response')
# plt.show()

E_freq_sim = np.fft.rfft(solver.recorded_Ez)*dt
source_freq = np.fft.rfft(solver.applied_source)*dt
omega = 2*np.pi*np.fft.rfftfreq(len(solver.recorded_Ez), dt)
E_freq_ana = -J0*omega/4*hankel2(0, omega/c*np.sqrt((xr-xs)**2+(yr-ys)**2))
E_freq_ana[0] = 0

# Restrict to bandwidth of the source
E_max = np.max(np.abs(source_freq))
mask = (np.abs(source_freq) > 0.005*E_max)

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