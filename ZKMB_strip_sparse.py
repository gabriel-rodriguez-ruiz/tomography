#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 17:11:05 2025

@author: gabriel
"""

from ZKMBsuperconductor import ZKMBSuperconductorKY
import numpy as np
import matplotlib.pyplot as plt
import scipy
from functions import get_components
from pauli_matrices import tau_y, sigma_y
from junction import Junction

L_x = 2000
t = 10
Delta_0 = 0.2#t/5     
Delta_1 = 0#t/20
Lambda = 0.56
phi_angle = 0
theta = np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -4*t#-2*t
t_J = t/2  #t/2    #t/2#t/5
phi_values = [0]#np.linspace(0, 2*np.pi, 240)    #240
k_y_values = [-2*np.pi/100]#np.array([-2*np.pi/100, 0, 2*np.pi/100]) #np.linspace(0, 2*np.pi, 200)  #200
antiparallel = False
k = 8

params = {"L_x":L_x, "t":t, "t_J":t_J,
          "Delta_0":Delta_0,
          "Delta_1":Delta_1,
          "mu":mu, "phi_values":phi_values,
          "k_y_values": k_y_values,
          "B": B, "phi_angle": phi_angle,
          "theta": theta, "antiparallel": antiparallel
          }

eigenvalues_k_phi = np.zeros((len(k_y_values), len(phi_values), k))
eigenvectors_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x, k), dtype=complex)
for i, k_y in enumerate(k_y_values):
    print(k_y)
    for j, phi in enumerate(phi_values):
        phi = np.array([phi])   #array of length 1
        S_ZKMB = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x, B_y, B_z)
        S_ZKMB2 = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x=(1-2*antiparallel)*B_x, B_y=(1-2*antiparallel)*B_y, B_z=(1-2*antiparallel)*B_z)
        J = Junction(S_ZKMB, S_ZKMB2, t_J, phi)
        eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(J.matrix, k=k, sigma=0) 
        eigenvalues_k_phi[i, j, :] = eigenvalues_sparse
        eigenvectors_k_phi[i, j, :, :] = eigenvectors_sparse

#%% Probability density
# eigenvectors_sparse = 1/2 * (eigenvectors_sparse - eigenvectors_sparse.conj())


index = np.arange(k)   #which zero mode (less than k)
probability_density = []
zero_state = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:,i], 2*L_x, L_y=1)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
# for i in index:
#     destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors[:,i], L_x, L_y)
#     probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
#     zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components

index = 1

fig, ax = plt.subplots()
ax.plot(probability_density[index][0, :]) 
ax.set_xlabel("x")
ax.set_ylabel(r"$|\Psi(x)|^2$")
ax.set_title("Probability density" + "\n" +
             f"E={np.format_float_scientific(eigenvalues_sparse[index], precision=2)}"
             + r"; $\mu=$" + f"{mu}"
            +r"; $\lambda=$" + f"{Lambda:.2}"
            +r"; $\Delta_0=$" + f"{Delta_0}"
            +r"; $\Delta_1=$" + f"{Delta_1}"
            +r"; $w_0=$"+f"{t}" + "\n"
            +r"$B_x=$"+f"{np.round(B_x, 2)}"
            +r"; $B_y=$"+f"{np.round(B_y, 2)}"
            +r"; $B_z=$"+f"{np.round(B_z, 2)}"
            + r"; $k_y=$" + f"{np.round(k_y,3)}")

# ax.set_title(f"{k for k in superconductor_params.keys()}")
plt.tight_layout()
plt.show()