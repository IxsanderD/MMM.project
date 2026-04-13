import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, mu_0
from Class_FCI import FCI
import time

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

xs = Nx//2
ys = Ny//2

solver=FCI(Nx,Ny,Nt,dx,dy,dt,eps,mu,k_max,sigma_max)
solver.add_material(3*Nx//5,4*Nx//5,3*Ny//5,4*Ny//5,2,2,1)
solver.construct_update_matrix()
end=time.perf_counter()
print(f"Runtime: {end - start:.6f} seconds")
solver.add_source(xs,ys,J0,tc,width)
solver.add_recorder(xs,ys)
# solver.update_loop()
# solver.show_recorder()
solver.animate()

import plotly.express as px

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