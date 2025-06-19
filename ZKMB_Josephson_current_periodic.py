#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  8 15:30:59 2025

@author: gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductorKY
from functions import get_components       
from junction import PeriodicJunctionInX, PeriodicJunctionInXWithQD
import scipy
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False})

L_x = 200
# L_y = 200
# L_x = L_y
t = 10
Delta_0 = 0.2#t/5     
Delta_1 = 0#t/20
Lambda = 0.56
phi_angle = 0  #np.pi/8 #np.pi/16#np.pi/2
theta = np.pi/2 #np.pi/2   #np.pi/2
B = 1/2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -3.8*t#-2*t
t_J = t    #t/2#t/5
# phi_values = np.sort(np.append(np.linspace(0, 2*np.pi, 50), np.pi))    #240
phi_values = np.linspace(0, 2*np.pi, 100)    #240
k_y_values = [0]#np.linspace(0, np.pi/10, 20)   #np.array([-2*np.pi/100, 0, 2*np.pi/100]) #np.linspace(0, 2*np.pi, 200)  #200
# k_x_values = np.array([0])
antiparallel = False

params = {"L_x":L_x, "t":t, "t_J":t_J,
          "Delta_0":Delta_0,
          "Delta_1":Delta_1,
          "mu":mu, "phi_values":phi_values,
          "k_y_values": k_y_values,
          "B": B, "phi_angle": phi_angle,
          "theta": theta, "antiparallel": antiparallel
          }

# params = {"L_y":L_y, "t":t, "t_J":t_J,
#           "Delta_0":Delta_0,
#           "Delta_1":Delta_1,
#           "mu":mu, "phi_values":phi_values,
#           "k_x_values": k_x_values,
#           "B": B, "phi_angle": phi_angle,
#           "theta": theta, "antiparallel": antiparallel
#           }

eigenvalues_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x))
eigenvectors_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x, 2*4*L_x), dtype=complex)
# eigenvalues_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x + 4))
# eigenvectors_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x + 4, 2*4*L_x + 4), dtype=complex)



for i, k_y in enumerate(k_y_values):
    print(k_y)
    for j, phi in enumerate(phi_values):
        phi = np.array([phi])   #array of length 1
        S_ZKMB = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x, B_y, B_z)
        S_ZKMB2 = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x=(1-2*antiparallel)*B_x, B_y=(1-2*antiparallel)*B_y, B_z=(1-2*antiparallel)*B_z)
        J = PeriodicJunctionInX(S_ZKMB, S_ZKMB2, t_J, phi)
        # J = PeriodicJunctionInXWithQD(S_ZKMB, S_ZKMB2, t_J, phi)
        eigenvalues, eigenvectors = np.linalg.eigh(J.matrix.toarray())
        eigenvalues_k_phi[i, j, :] = eigenvalues
        eigenvectors_k_phi[i, j, :, :] = eigenvectors

#%%
total_energy_k = np.zeros((len(k_y_values), len(phi_values)))

for i in range(len(k_y_values)):
    E_positive = np.where(eigenvalues_k_phi[i, :, :] > 0, eigenvalues_k_phi, np.zeros_like(eigenvalues_k_phi))
    total_energy_k[i, :] = np.sum(E_positive[i, :, :], axis=1)

dphi = np.diff(phi_values)
Josephson_current_k_eigenvalues = np.gradient(-total_energy_k, dphi[0], axis=1)
Josephson_current_eigenvalues = np.sum(Josephson_current_k_eigenvalues, axis=0)

fig, ax = plt.subplots()
ax.plot(phi_values, total_energy_k[0, :], "ok", markersize=0.5)
ax.legend()
plt.show()

fig, ax = plt.subplots()
for i in range(len(k_y_values)):
    ax.scatter(phi_values/(2*np.pi), Josephson_current_k_eigenvalues[i,:],
           marker=".",
           label=r"$|k_y|=$" + f"{np.abs(np.round(k_y_values[i], 3))}")
ax.legend()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")   
ax.set_title("Josephson current for given k\n"+
             r"$\theta=$" + f"{np.round(theta, 2)}"+
             r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
             r"; $B=$" + f"{B}")

fig, ax = plt.subplots()

ax.scatter(phi_values/(2*np.pi), Josephson_current_eigenvalues,
       marker=".",
       label=r"Total")
ax.legend()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")   
ax.set_title("Josephson current for given k\n"+
             r"$\theta=$" + f"{np.round(theta, 2)}"+
             r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
             r"; $B=$" + f"{B}")
plt.show()