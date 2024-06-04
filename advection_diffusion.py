import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
from jax import grad, jit

import matplotlib.pyplot as plt
import timeit

# Philip Mocz (2024)
# Solve the advection-diffusion equation using a finite difference method
# Plug it into an optimization problem to find the wind parameters that maximize pollution
# Use either Nelder-Mead or autodiff (JAX) to solve the optimization problem

# Global variables
diffusivity = 0.05
t_stop = 0.25
N = 81
M = 50
dx = 1.0 / (N-1)
dt = t_stop / M
t = np.linspace(0, t_stop, M+1)


# === Numpy version of the simulation ========================================

def index_function(i, j, N):
  # maps index (i,j) to the vector index in our solution vector 
  # (the grid size is N^2)
  return j*N + i


def initial_condition(x, y):
  # initial condition for the pollution
  return 2.0*np.exp(-100.0*((x-0.25)**2+(y-0.25)**2))+np.exp(-150.0*((x-0.65)**2+(y-0.4)**2))


def do_simulation(x):
  # Solve the advection-diffusion equation using a finite difference method
  # Keep track of the pollution
  W = x[0]
  theta = x[1]

  # Construct the matrix (D) for the linear system to be solved at each time step
  D = np.eye(N**2, N**2)
  for i in range(1, N-1):
    for j in range(1, N-1):
      D[index_function(i,j,N),index_function(i,j,N)]   = dt*(1.0/dt + 4.0*diffusivity/dx**2)
      D[index_function(i,j,N),index_function(i+1,j,N)] = dt*( W*np.cos(theta)/(2.0*dx) - diffusivity/dx**2)
      D[index_function(i,j,N),index_function(i-1,j,N)] = dt*(-W*np.cos(theta)/(2.0*dx) - diffusivity/dx**2)
      D[index_function(i,j,N),index_function(i,j+1,N)] = dt*( W*np.sin(theta)/(2.0*dx) - diffusivity/dx**2)
      D[index_function(i,j,N),index_function(i,j-1,N)] = dt*(-W*np.sin(theta)/(2.0*dx) - diffusivity/dx**2)

  D = csc_matrix(D)  # sparse representation of the matrix
  B = splu(D)        # do an LU decomposition of the matrix

  # Construct initial (t=0) solution
  xlin = np.linspace(0.0, 1.0, N)
  U = np.zeros(N**2)
  for i in range(1, N-1):
    for j in range(1, N-1):
      U[index_function(i,j,N)] = initial_condition(xlin[i], xlin[j])

  # Keep track of pollution
  pollution = np.zeros(M+1)
  pollution[0] = U[index_function(N//2+1,N//2+1,N)]

  # Solve for the time evolution
  for i in range(M):
    U = B.solve(U)
    pollution[i+1] = U[index_function(N//2+1,N//2+1,N)]
  
  pollution[M] = U[index_function(N//2+1,N//2+1,N)]

  pollution_total = np.trapz(pollution, t)

  return U, pollution, pollution_total


def loss(x, info):
  # loss function that wraps the simulation
  _, _, pollution_total = do_simulation(x)

  # display information
  print('{0:4d}   {1: 3.6f}   {2: 3.6f} {3: 3.6f}'.format(info['Nfeval'], W, theta, pollution_total))
  info['Nfeval'] += 1

  return -pollution_total


# === JAX version of the simulation ==========================================

@jit
def initial_condition_jax(x, y):
  # initial condition for the pollution -- JAX version
  return 2.0*jnp.exp(-100.0*((x-0.25)**2+(y-0.25)**2))+jnp.exp(-150.0*((x-0.65)**2+(y-0.4)**2))


@jit
def do_simulation_jax(x):
  # Solve the advection-diffusion equation with finite difference -- JAX version
  # Keep track of the pollution

  W = x[0]
  theta = x[1]

  # Construct the matrix (D) for the linear system to be solved at each time step
  main_diag = jnp.ones(N**2)  * dt*(1.0/dt + 4.0*diffusivity/dx**2)
  off_diag1 = jnp.ones(N**2-1)* dt*( W*jnp.cos(theta)/(2.0*dx) - diffusivity/dx**2)
  off_diag2 = jnp.ones(N**2-1)* dt*(-W*jnp.cos(theta)/(2.0*dx) - diffusivity/dx**2)
  off_diag3 = jnp.ones(N**2-N)* dt*( W*jnp.sin(theta)/(2.0*dx) - diffusivity/dx**2)
  off_diag4 = jnp.ones(N**2-N)* dt*(-W*jnp.sin(theta)/(2.0*dx) - diffusivity/dx**2)
  D = jnp.diag(main_diag) + jnp.diag(off_diag1, 1) + jnp.diag(off_diag2, -1) + jnp.diag(off_diag3, N) + jnp.diag(off_diag4, -N)
  bndry1 = jnp.arange(N)
  bndry2 = (N-1)*N + jnp.arange(N)
  bndry3 = jnp.arange(N)*N
  bndry4 = jnp.arange(N)*N + N-1
  bndry = jnp.concatenate((bndry1, bndry2, bndry3, bndry4))
  D = D.at[bndry, :].set(0.0)
  D = D.at[bndry, bndry].set(1.0)

  B = jax.scipy.linalg.lu_factor(D)  # do an LU decomposition of the matrix

  # Construct initial (t=0) solution
  x = jnp.linspace(0.0, 1.0, N)
  X, Y = jnp.meshgrid(x, x)
  U = initial_condition_jax(X, Y)
  U = U.at[0,:].set(0.0)
  U = U.at[-1,:].set(0.0)
  U = U.at[:,0].set(0.0)
  U = U.at[:,-1].set(0.0)
  U = U.flatten()

  # Keep track of pollution
  ctr = (N//2+1)*N + N//2+1
  pollution = jnp.zeros(M+1)
  pollution = pollution.at[0].set(U[ctr])

  # Solve for the time evolution
  for i in range(M):
    U = jax.scipy.linalg.lu_solve(B, U)
    pollution = pollution.at[i+1].set(U[ctr])

  pollution = pollution.at[M].set(U[ctr])

  t = jnp.linspace(0, t_stop, M+1)
  pollution_total = jnp.trapezoid(pollution, t)

  return U, pollution, pollution_total


@jit
def loss_jax(x):
  # loss function that wraps the simulation (jax version)
  _, _, pollution_total = do_simulation_jax(x)

  return -pollution_total


# === Main ==================================================================

def main():

  # Wind parameters (initial guess)
  W = 1.0
  theta = np.pi/2.0
  
  # Optimize the wind parameters to find which values maximize the pollution
  x0 = np.array([W, theta])
  bounds = [(0, 3), (0, np.pi)]
  start = timeit.default_timer()
  ###wind = minimize(loss, x0, args=({'Nfeval':0},), method='Nelder-Mead', tol=1e-8, bounds=bounds)
  print("Optimization process took:", timeit.default_timer() - start, "seconds")
  ###W = wind.x[0]
  ###theta = wind.x[1]
  print('Optimized wind parameters:', W, theta)

  # Carry out simulation with the optimized parameters
  start = timeit.default_timer()
  U, pollution, pollution_total = do_simulation_jax(x0)
  print("JAX took:", timeit.default_timer() - start, "seconds")
  start = timeit.default_timer()
  U, pollution, pollution_total = do_simulation(x0)
  print("Numpy took:", timeit.default_timer() - start, "seconds")

  # XXX
  start = timeit.default_timer()
  grad_sim = grad(loss_jax)
  print(grad_sim(x0))
  print("XXX took:", timeit.default_timer() - start, "seconds")

  # Print the level of pollution
  print('Total pollution:', pollution_total)

  # Plot the pollution as a function of time
  fig = plt.figure(figsize=(4,4), dpi=120)
  plt.plot(t, pollution, 'b-')
  plt.xlabel('Time')
  plt.ylabel('Pollution')
  plt.xlim(0, t_stop)
  plt.ylim(0.0, 0.5)
  plt.show()

  # Plot the solution
  fig = plt.figure(figsize=(4,4), dpi=120)
  U_plot = np.zeros((N, N))
  for i in range(N):
    for j in range(N):
      U_plot[j, i] = U[index_function(i, j, N)]

  plt.imshow(U_plot, cmap='Purples')
  plt.clim(0.0, 0.4)
  plt.contour(U_plot, levels=10, colors='black', alpha=0.5)
  plt.plot(0.5*N, 0.5*N, 'bs', markersize=8)
  ax = plt.gca()
  ax.invert_yaxis()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)	
  ax.set_aspect('equal')

  # Save figure
  plt.savefig('simulation.png',dpi=240)
  plt.show()

  return 0


if __name__== "__main__":
  main()
