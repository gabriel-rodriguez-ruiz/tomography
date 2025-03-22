#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 10:34:43 2025

@author: gabriel
"""

from ZKMBsuperconductor import ZKMBSuperconductorKY
import numpy as np
import matplotlib.pyplot as plt
import scipy
from functions import get_components
from pauli_matrices import tau_y, sigma_y

L_x = 1000
t = 10
Delta_0 = 0.2
Delta_1 = 0
Lambda = 0.56
theta = np.pi/2     #spherical coordinates
phi = 0
B = 2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi)
B_y = B * np.sin(theta) * np.sin(phi)
B_z = B * np.cos(theta)
mu = -40   #in the middle ot the topological phase
k = 16
k_y = 0#2*np.pi/100
H = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1, Lambda,
                             B_x, B_y, B_z)
# H = ZKMBSuperconductor(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
#                              B_x, B_y, B_z)
# H = ZKMBSparseSuperconductorPeriodicInY(L_x, L_y, t, mu, Delta_0, Delta_1, Lambda,
#                              B_x, B_y, B_z)

# eigenvalues_sparse, eigenvectors_sparse = scipy.sparse.linalg.eigsh(H.matrix, k=k, sigma=0) 
eigenvalues_sparse, eigenvectors_sparse = np.linalg.eigh(H.matrix) 

# C = np.kron(tau_y, sigma_y)     #charge conjugation operator
# M = np.kron(np.eye(L_x*L_y), C)
# U = eigenvectors_sparse
# D = np.linalg.inv(U) @ H.matrix @ U
# C_prime = U @ M @ np.linalg.inv(U)

superconductor_params = {"t":t, "Delta_0":Delta_0,
          "mu":mu, "Delta_1":Delta_1,
          "B_x":B_x, "B_y":B_y, "B_z":B_z,
          "Lambda":Lambda,
          }

#%% Probability density
# eigenvectors_sparse = 1/2 * (eigenvectors_sparse - eigenvectors_sparse.conj())


index = 0

destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_sparse[:, 4*L_x//2-1+index], L_x, L_y=1)
probability_density = (np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2))

# for i in index:
#     destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors[:,i], L_x, L_y)
#     probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
#     zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components


fig, ax = plt.subplots()
ax.plot(probability_density) 
ax.set_xlabel("x")
ax.set_ylabel(r"$|\Psi(x)|^2$")
ax.set_title("Probability density" + "\n" +
             f"E={np.format_float_scientific(eigenvalues_sparse[4*L_x//2-1+index], precision=2)}"
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