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
phi_values = np.sort(np.append(np.linspace(0, 2*np.pi, 50), np.pi))    #240
k_x_values = np.linspace(-0.1*np.pi, 0.1*np.pi, 100)#np.linspace(-0.2*np.pi, 0.2*np.pi, 50)#np.linspace(-0.1*np.pi, 0.1*np.pi, 50)
k_y_values = np.linspace(-0.1*np.pi, 0.1*np.pi, 100)#np.linspace(-0.2*np.pi, 0.2*np.pi, 50)#np.linspace(-0.1*np.pi, 0.1*np.pi, 50)  #np.linspace(0, 2*np.pi, 10)   #np.array([-2*np.pi/100, 0, 2*np.pi/100]) #np.linspace(0, 2*np.pi, 200)  #200
antiparallel = False

params = {"L_x":L_x, "t":t, 
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
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d', computed_zorder=False)
plt.rcParams.update({
    "text.usetex": True})
# Make data.
X = k_x_values/np.pi
Y = k_y_values/np.pi
Y, X = np.meshgrid(X, Y)
Z = eigenvalues[:, :, 2]

import matplotlib.colors as colors
palette = plt.cm.gray.with_extremes(over='orange', under="blue", bad='b')
palette2 = plt.cm.gray.with_extremes(over='green', under='red', bad='b')

bounds = [-2, 0, 2] # Boundaries for negative, zero, and positive

# Plot the surface.
# z = 2*eigenvalues[:, :, 0]/Delta_0
# z[z<-6] = np.nan
# ax.plot_surface(X, Y, z,
#                 linewidth=0, antialiased=True,
#                 alpha=0.15, cmap=palette, vmin=-1, vmax=1)
ax.plot_surface(X, Y, eigenvalues[:, :, 0]/Delta_0 - 1.5,
                linewidth=0, antialiased=True,
                alpha=0.15, cmap=palette, vmin=-1, vmax=1)

ax.plot_surface(X, Y, eigenvalues[:, :, 1]/Delta_0,
                linewidth=0, antialiased=True,
                alpha=0.2, cmap=palette, norm=colors.Normalize(vmin=0, vmax=0.01), edgecolor='blue', lw=0.5, rstride=10, cstride=10)

ax.plot_surface(X, Y, eigenvalues[:, :, 1]/Delta_0,
                linewidth=0, antialiased=True,
                alpha=0.2, cmap=palette, norm=colors.Normalize(vmin=0, vmax=0.01))

# ax.plot_surface(X, Y, 2*eigenvalues[:, :, 1]/Delta_0,
#                 linewidth=0, antialiased=True,
#                 alpha=0.3, zorder=1)
# ax.plot_surface(X, Y, 2*eigenvalues[:, :, 1]/Delta_0,
#                 linewidth=0, antialiased=True,
#                 alpha=0.4, cmap=palette, norm=colors.Normalize(vmin=0, vmax=0.01))
ax.plot_surface(X, Y, eigenvalues[:, :, 2]/Delta_0,
                linewidth=0, antialiased=True,
                alpha=0.15, cmap=palette2, norm=colors.Normalize(vmin=-0.01, vmax=0))
ax.plot_surface(X, Y, eigenvalues[:, :, 2]/Delta_0,
                linewidth=0, antialiased=True,
                alpha=0.2, cmap=palette2, norm=colors.Normalize(vmin=-0.01, vmax=0), edgecolor='green', lw=0.5, rstride=10, cstride=10)

ax.plot_surface(X, Y, eigenvalues[:, :, 3]/Delta_0 + 2.5,
                linewidth=0, antialiased=True,
                alpha=0.15, cmap=palette2, vmin=-1, vmax=1)

ax.set(xlim=(min(1.3*k_y_values/np.pi), max(1.3*k_y_values/np.pi)), ylim=(min(1.3*k_x_values/np.pi), max(1.3*k_x_values/np.pi)), zlim=(-6, 6), #zlim=(-15, 15),
       xlabel='X', ylabel='Y', zlabel='Z')

# plt.contour(X, Y, eigenvalues[:, :, 0]/Delta_0-2, levels=np.array([0]), colors=["k"])
ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, levels=np.array([0]), colors=["k"], zorder=2)
ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, levels=np.array([0]), colors=["k"])
# plt.contour(X, Y, eigenvalues[:, :, 3]/Delta_0+1, levels=np.array([0]), colors=["k"])
z = 2*eigenvalues[:, :, 3]/Delta_0 + 1
z[z>6] = np.nan
ax.contour(X, Y, z, zdir='x', offset=ax.get_xlim()[0], colors='green', levels=np.array([0]),
           alpha=0.6)
ax.contour(X, Y, z, zdir='y', offset=ax.get_ylim()[1], colors='green', levels=np.array([0]),
           alpha=0.6)

# ax.contour(X, Y, 2*eigenvalues[:, :, 3]/Delta_0+1, zdir='x', offset=ax.get_xlim()[0], colors='green', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, zdir='x', offset=ax.get_xlim()[0], colors='green', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, zdir='y', offset=ax.get_ylim()[1], colors='green', levels=np.array([0]))
# ax.contour(X, Y, 2*eigenvalues[:, :, 0]/Delta_0-1, zdir='x', offset=ax.get_xlim()[0], colors='blue', levels=np.array([0]))
z = 2*eigenvalues[:, :, 0]/Delta_0 -1  ##########
z[z<-6] = np.nan
ax.contour(X, Y, z, zdir='x', offset=ax.get_xlim()[0], colors='blue', levels=np.array([0]),
           alpha=0.5)
ax.contour(X, Y, z, zdir='y', offset=ax.get_ylim()[1], colors='blue', levels=np.array([0]),
           alpha=0.5)

# ax.contour(X, Y, 2*eigenvalues[:, :, 3]/Delta_0, zdir='x', offset=ax.get_xlim()[0], colors='blue', levels=np.array([0]))
# ax.contour(X, Y, 2*eigenvalues[:, :, 3]/Delta_0, zdir='y', offset=ax.get_ylim()[1], colors='blue', levels=np.array([0]))


ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, zdir='x', offset=ax.get_xlim()[0], colors='blue', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, zdir='y', offset=ax.get_ylim()[1], colors='blue', levels=np.array([0]))
# ax.contour(X, Y, 2*eigenvalues[:, :, 0]/Delta_0 -1, zdir='y', offset=ax.get_ylim()[1], colors='blue', levels=np.array([0]))

# ax.contour(X, Y, 2*eigenvalues[:, :, 3]/Delta_0 +1, zdir='y', offset=ax.get_ylim()[1], colors='green', levels=np.array([0]))

XX = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
YY = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 100)
YY, XX = np.meshgrid(XX, YY)
ax.contour(XX, YY, np.zeros_like(2*eigenvalues[:, :, 2]/Delta_0), zdir='y', offset=ax.get_ylim()[1], levels=np.array([0]), colors="k", linestyles='dashed')
ax.contour(XX, YY, np.zeros_like(2*eigenvalues[:, :, 2]/Delta_0), zdir='x', offset=ax.get_xlim()[0], levels=np.array([0]), colors="k", linestyles='dashed')
# ax.contour(X, Y, np.zeros_like(eigenvalues[:, :, 1]), zdir='z', offset=ax.get_zlim()[0], levels=10, colors="k", linestyles='dashed')

# ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 1]/Delta_0), zdir='y', offset=ax.get_ylim()[1], levels=np.array([0]), colors="k", linestyles='dashed')
# ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 1]/Delta_0), zdir='x', offset=ax.get_xlim()[0], levels=np.array([0]), colors="k", linestyles='dashed')

# ax.contour(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set_xlabel(r'$k_x/\pi$', fontsize=18, labelpad=-5)
ax.set_ylabel(r'$k_y/\pi$', fontsize=18, labelpad=-4)
# ax.set_zlabel(r'$E/\Delta_0$', fontsize=18, labelpad=-9)
ax.set_zlabel("")
ax.text(0.1, 0.1, 7.3, r"$E/\Delta_0$", fontsize=18)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
ax.tick_params(pad=-3)

ax.fill_between(-0.13, np.linspace(-0.1, 0.1, 100), 2*eigenvalues[50, :, 1]/Delta_0, -0.13, np.linspace(-0.1, 0.1, 100), 0,
                where=2*eigenvalues[50, :, 1]/Delta_0>0,
                alpha=0.5, color="orange")

ax.fill_between(-0.13, np.linspace(-0.1, 0.1, 100), 2*eigenvalues[50, :, 2]/Delta_0, -0.13, np.linspace(-0.1, 0.1, 100), 0,
                where=2*eigenvalues[50, :, 2]/Delta_0<0,
                alpha=0.5, color="red")


# ax.axes.set_zlim3d(bottom=-6, top=6)
# ax.axes.set_xlim3d((np.float64(-0.11458333333333334), np.float64(0.11458333333333334)))
# ax.axes.set_ylim3d((np.float64(-0.11458333333333334), np.float64(0.11458333333333334))
# )

# ax.axes.set_xlim3d(left=-0.1/np.pi, right=0.1/np.pi) 
# ax.axes.set_ylim3d(left=-0.1/np.pi, right=0.1/np.pi) 
# ax.axes.set_zlim3d(bottom=-6, top=6) 

# z = 2*eigenvalues[0, 25, 1]/Delta_0
# ax.plot(0.1*np.ones(50), np.linspace(0, ax.get_ylim()[1]), z*np.ones(50),
#         alpha=0.5, linestyle=":", color="blue")
# ax.plot(0.1*np.ones(50), 0*np.ones(50), np.linspace(-6, z),
#         alpha=0.5, linestyle=":", color="blue")

# z = 2*eigenvalues[25, 0, 1]/Delta_0
# ax.plot(np.linspace(0, ax.get_ylim()[0]), -0.1*np.ones(50), z*np.ones(50),
#         alpha=0.5, linestyle=":", color="blue")
# ax.plot(0*np.ones(50), -0.1*np.ones(50), np.linspace(-6, z),
#         alpha=0.5, linestyle=":", color="blue")

# ax.plot(np.linspace(-0.1, 0.1), 0*np.ones(50), 2*eigenvalues[:, 25, 1]/Delta_0,
#         alpha=0.2, color="blue", linewidth=2)
# ax.plot(0*np.ones(50), np.linspace(-0.1, 0.1), 2*eigenvalues[25, :, 1]/Delta_0,
#         alpha=0.2, color="blue", linewidth=2)

# ax.plot(np.linspace(-0.1, 0.1), 0*np.ones(50), 2*eigenvalues[:, 24, 2]/Delta_0,
#         alpha=0.3, color="green", linewidth=2)
# ax.plot(0*np.ones(50), np.linspace(-0.1, 0.1), 2*eigenvalues[24, :, 2]/Delta_0,
#         alpha=0.3, color="green", linewidth=2)

plt.tight_layout()

plt.show()
#%%

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

ax.contour(X, Y, 2*eigenvalues[:, :, 0]/Delta_0 -1, zdir='x', offset=0, colors='black', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 0]/Delta_0 -1, zdir='y', offset=0, colors='black', levels=np.array([0]))


ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, zdir='x', offset=0, colors='red', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, zdir='y', offset=0, colors='red', levels=np.array([0]))

ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, zdir='x', offset=0, colors='blue', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, zdir='y', offset=0, colors='blue', levels=np.array([0]))

ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 2]/Delta_0), zdir='y', offset=0, levels=np.array([0]), colors="k", linestyles='dashed')
ax.contour(X, Y, np.zeros_like(2*eigenvalues[:, :, 2]/Delta_0), zdir='x', offset=0, levels=np.array([0]), colors="k", linestyles='dashed')

ax.contour(X, Y, 2*eigenvalues[:, :, 3]/Delta_0 +1, zdir='x', offset=0, colors='black', levels=np.array([0]))
ax.contour(X, Y, 2*eigenvalues[:, :, 3]/Delta_0 +1, zdir='y', offset=0, colors='black', levels=np.array([0]))

ax.contourf(X, Y, 2*eigenvalues[:, :, 1]/Delta_0, levels=np.array([0,1]), colors=["r"],
           linestyles="--", offset=-6, alpha=0.3)
ax.contourf(X, Y, 2*eigenvalues[:, :, 2]/Delta_0, levels=np.array([-1, 0]), colors=["b"],
           linestyles="--", offset=-6, alpha=0.3)

ax.axes.set_zlim3d(bottom=-6, top=6)
ax.axes.set_xlim3d((np.float64(-0.11458333333333334), np.float64(0.11458333333333334)))
ax.axes.set_ylim3d((np.float64(-0.11458333333333334), np.float64(0.11458333333333334))
)

ax.fill_between(np.linspace(-0.1, 0.1, 50), 0, 0, np.linspace(-0.1, 0.1, 50), 0, 2*eigenvalues[25, :, 1]/Delta_0,
                where=2*eigenvalues[25, :, 1]/Delta_0>0,
                alpha=0.3, color="r")
ax.fill_between(np.linspace(-0.1, 0.1, 50), 0, 0, np.linspace(-0.1, 0.1, 50), 0, 2*eigenvalues[25, :, 2]/Delta_0,
                where=2*eigenvalues[25, :, 2]/Delta_0<0,
                alpha=0.3, color="b")
ax.set_box_aspect(aspect=(1,1,1))

ax.set_xlabel(r'$k_y/\pi$')
ax.set_ylabel(r'$k_x/\pi$')
ax.set_zlabel(r'$E(k_x, k_y)/\Delta$')


# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])