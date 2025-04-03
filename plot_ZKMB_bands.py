#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:25:03 2024

@author: gabriel
"""

from ZKMBsuperconductor import ZKMBSuperconductorKX, ZKMBSuperconductorKY
import numpy as np
import matplotlib.pyplot as plt


# L_y = 200
# k_x_values = np.linspace(-np.pi/6, np.pi/6, 100)

L_x = 400
k_y_values = np.linspace(-np.pi/6, np.pi/6, 100)

t = 10   #10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0
B = 6*Delta_0    #0.4*Delta_0    #2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -39  #in the middle ot the topological phase

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

#%% Bands

E_k_y = np.zeros((len(k_y_values), 4*L_x))

for i, k_y in enumerate(k_y_values):
    S = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1, Lambda,
                               B_x, B_y, B_z)
    E_k_y[i, :] = np.linalg.eigvalsh(S.matrix)

fig, ax = plt.subplots()

ax.plot(k_y_values, E_k_y/Delta_0, color="black")
ax.plot(k_y_values[33:66], E_k_y[33:66,2*L_x-1:2*L_x+1]/Delta_0, color="red")

ax.set_ylim((-2, 2))
ax.set_xlim((-np.pi/6, np.pi/6))

ax.set_xlabel(r"$k_y$")
ax.set_ylabel(r"$\frac{E(k_y)}{\Delta_0}$")
ax.set_title(r"$\mu=$"+f"{np.round(S.mu, 2)}")
fig.suptitle(r"$L_x=$"+f"{L_x}"
             +r"; $\lambda=$" + f"{S.Lambda:.2}"
             +r"; $\Delta_0=$" + f"{S.Delta_0}"
             +r"; $\Delta_1=$" + f"{S.Delta_1}"
             +r"; $w_0=$"+f"{S.t}" + "\n"
             +r"$B_x=$"+f"{np.round(B_x, 2)}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}"
             +r"; $B_z=$"+f"{np.round(B_z, 2)}")

plt.tight_layout()

plt.show()

#%% Bands
# E_k_x = np.zeros((len(k_x_values), 4*L_y))

# for i, k_x in enumerate(k_x_values):
#     S = ZKMBSuperconductorKX(k_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
#                                B_x, B_y, B_z)
#     E_k_x[i, :] = np.linalg.eigvalsh(S.matrix)
    
E_k_y = np.zeros((len(k_y_values), 4*L_x))

for i, k_y in enumerate(k_y_values):
    S = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1, Lambda,
                               B_x, B_y, B_z)
    E_k_y[i, :] = np.linalg.eigvalsh(S.matrix)
            
# fig, axs = plt.subplots(1, 2)
# axs[0].plot(k_x_values, E_k_x/Delta_0, color="black")
# axs[0].plot(k_x_values[10:90], E_k_x[10:90,2*L_y-1:2*L_y+1]/Delta_0, color="red")

# axs[0].set_xlabel(r"$k_x$")
# axs[0].set_ylabel(r"$\frac{E(k_x)}{\Delta_0}$")
# axs[0].set_title(r"$\mu=$"+f"{np.round(S.mu, 2)}")
# fig.suptitle(r"$L_y=$"+f"{L_y}"
#              +r"; $\lambda=$" + f"{S.Lambda:.2}"
#              +r"; $\Delta_0=$" + f"{S.Delta_0}"
#              +r"; $\Delta_1=$" + f"{S.Delta_1}"
#              +r"; $w_0=$"+f"{S.t}" + "\n"
#              +r"$B_x=$"+f"{np.round(B_x, 2)}"
#              +r"; $B_y=$"+f"{np.round(B_y, 2)}"
#              +r"; $B_z=$"+f"{np.round(B_z, 2)}")
# axs[0].set_ylim((-2, 2))

fig, axs = plt.subplots(1, 2)

axs[0].plot(k_y_values, E_k_y/Delta_0, color="black")
axs[0].plot(k_y_values[5:95], E_k_y[5:95,2*L_x-1:2*L_x+1]/Delta_0, color="red")

axs[0].set_xlabel(r"$k_y$")
axs[0].set_ylabel(r"$\frac{E(k_y)}{\Delta_0}$")
axs[0].set_title(r"$\mu=$"+f"{np.round(S.mu, 2)}")
fig.suptitle(r"$L_x=$"+f"{L_x}"
             +r"; $\lambda=$" + f"{S.Lambda:.2}"
             +r"; $\Delta_0=$" + f"{S.Delta_0}"
             +r"; $\Delta_1=$" + f"{S.Delta_1}"
             +r"; $w_0=$"+f"{S.t}" + "\n"
             +r"$B_x=$"+f"{np.round(B_x, 2)}"
             +r"; $B_y=$"+f"{np.round(B_y, 2)}"
             +r"; $B_z=$"+f"{np.round(B_z, 2)}")
axs[0].set_ylim((-2, 2))

plt.tight_layout()

plt.show()

#%%

k_y = 0
mu_values = np.linspace(-4*t - 2* np.sqrt(B**2 - Delta_0**2), -4*t + 2 * np.sqrt(B**2 - Delta_0**2), 200)
E_mu = np.zeros((len(mu_values), 4*L_x))
for i, mu in enumerate(mu_values):
    S = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1, Lambda,
                               B_x, B_y, B_z)
    E_mu[i, :] = np.linalg.eigvalsh(S.matrix)

axs[1].plot(mu_values, E_mu/Delta_0, color="black")
axs[1].plot(mu_values[50:150], E_mu[50:150,2*L_x-1:2*L_x]/Delta_0, color="red")

axs[1].set_xlabel(r"$\mu$")
axs[1].set_ylabel(r"$\frac{E(\mu)}{\Delta_0}$")
axs[1].set_title(r"$k_y=$"+f"{k_y}")
axs[1].axvline(-4*t + np.sqrt(B**2 - Delta_0**2))
axs[1].axvline(-4*t - np.sqrt(B**2 - Delta_0**2))

# axs[1].axvline(-4*t + np.sqrt(B**2 - Delta_0**2))
# axs[1].axvline(-4*t - np.sqrt(B**2 - Delta_0**2))

axs[1].set_ylim((-2, 2))
plt.tight_layout()
plt.show()

