# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 17:58:50 2025

@author: Gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductorKXKY
from functions import get_components       
from junction import Junction, PeriodicJunction
import scipy
import matplotlib.pyplot as plt

L_x = 200
t = 10
Delta_0 = 0.2#t/5     
Delta_1 = 0#t/20
Lambda = 0.56
phi_angle = 0#np.pi/2
theta = np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -4*t#-2*t
t_J = t/2    #t/2#t/5
phi_values = np.sort(np.append(np.linspace(0, 2*np.pi, 50), np.pi))    #240
k_x_values = np.linspace(-0.1*np.pi, 0.1*np.pi, 50)#np.linspace(-0.2*np.pi, 0.2*np.pi, 50)#np.linspace(-0.1*np.pi, 0.1*np.pi, 50)
k_y_values = np.linspace(-0.1*np.pi, 0.1*np.pi, 50)#np.linspace(-0.2*np.pi, 0.2*np.pi, 50)#np.linspace(-0.1*np.pi, 0.1*np.pi, 50)  #np.linspace(0, 2*np.pi, 10)   #np.array([-2*np.pi/100, 0, 2*np.pi/100]) #np.linspace(0, 2*np.pi, 200)  #200
antiparallel = False

params = {"L_x":L_x, "t":t, "t_J":t_J,
          "Delta_0":Delta_0,
          "Delta_1":Delta_1,
          "mu":mu, "phi_values":phi_values,
          "k_y_values": k_y_values,
          "B": B, "phi_angle": phi_angle,
          "theta": theta, "antiparallel": antiparallel
          }


eigenvalues = np.zeros((len(k_x_values), len(k_y_values), 4))

for i, k_x in enumerate(k_x_values):
    for j, k_y in enumerate(k_y_values):     
        S = ZKMBSuperconductorKXKY(k_x, k_y, t, mu, Delta_0, Delta_1, Lambda,
                                   B_x,
                                   B_y, B_z)
        eigenvalues[i, j, :] = np.linalg.eigvalsh(S.matrix)
        
        
#%% Plot
from matplotlib import cm

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = k_x_values/np.pi
Y = k_y_values/np.pi
X, Y = np.meshgrid(X, Y)
Z = eigenvalues[:, :, 2]

# Plot the surface.
# ax.plot_surface(X, Y, eigenvalues[:, :, 0],
#                 linewidth=0, antialiased=False,
#                 alpha=0.2, cmap='PuOr', vmin=-1, vmax=1)
ax.plot_surface(X, Y, eigenvalues[:, :, 1]/Delta_0,
                linewidth=0, antialiased=True,
                alpha=0.3, cmap='PuOr', vmin=-1, vmax=1, edgecolor='red', lw=0.5, rstride=4, cstride=4)
ax.plot_surface(X, Y, eigenvalues[:, :, 2]/Delta_0,
                linewidth=0, antialiased=True,
                alpha=0.3, cmap='PuOr', vmin=-1, vmax=1, edgecolor='blue', lw=0.5, rstride=4, cstride=4)
# ax.plot_surface(X, Y, eigenvalues[:, :, 3],
#                 linewidth=0, antialiased=False,
#                 alpha=0.2, cmap='PuOr', vmin=-1, vmax=1)

ax.set(xlim=(min(1.3*k_y_values/np.pi), max(1.3*k_y_values/np.pi)), ylim=(min(1.3*k_x_values/np.pi), max(1.3*k_x_values/np.pi)), zlim=(-6, 6), #zlim=(-15, 15),
       xlabel='X', ylabel='Y', zlabel='Z')

# plt.contour(X, Y, eigenvalues[:, :, 0], levels=np.array([0]), colors=["k"])
ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, levels=np.array([0]), colors=["k"])
ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, levels=np.array([0]), colors=["k"])
# plt.contour(X, Y, eigenvalues[:, :, 3], levels=np.array([0]), colors=["k"])

ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, zdir='x', offset=ax.get_xlim()[0], colors='blue', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, zdir='y', offset=ax.get_ylim()[1], colors='blue', levels=np.array([0]))

ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, zdir='x', offset=ax.get_xlim()[0], colors='red', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, zdir='y', offset=ax.get_ylim()[1], colors='red', levels=np.array([0]))

ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 2]/Delta_0), zdir='y', offset=ax.get_ylim()[1], levels=np.array([0]), colors="k", linestyles='dashed')
ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 2]/Delta_0), zdir='x', offset=ax.get_xlim()[0], levels=np.array([0]), colors="k", linestyles='dashed')
# ax.contour(X, Y, np.zeros_like(eigenvalues[:, :, 1]), zdir='z', offset=ax.get_zlim()[0], levels=10, colors="k", linestyles='dashed')

ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 1]/Delta_0), zdir='y', offset=ax.get_ylim()[1], levels=np.array([0]), colors="k", linestyles='dashed')
ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 1]/Delta_0), zdir='x', offset=ax.get_xlim()[0], levels=np.array([0]), colors="k", linestyles='dashed')

# ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set_xlabel(r'$k_y/\pi$')
ax.set_ylabel(r'$k_x/\pi$')
ax.set_zlabel(r'$E(k_x, k_y)/\Delta$')

plt.tight_layout()

