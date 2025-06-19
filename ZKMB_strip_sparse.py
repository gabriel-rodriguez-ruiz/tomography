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

L_x = 300#2000
t = 10
Delta_0 = 0.2#t/5     
Delta_1 = 0#t/20
Lambda = 0.56
phi_angle = np.pi/8
theta = np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -4*t#-2*t
t_J = t/2  #t/2    #t/2#t/5
phi_values = [1.4]#np.linspace(0, 2*np.pi, 240)    #240
k_y_values = np.array([0]) #np.linspace(0, 2*np.pi, 200)  #200
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
from functions import mean_spin, mean_spin_xy


index = np.arange(k)   #which zero mode (less than k)
probability_density = []
edge_states = np.zeros((len(k_y_values), len(phi_values), 16, len(index)), dtype=complex)
spin_values = []

for i in index:
    destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors_k_phi[0, 0, :, i], 2*L_x, L_y=1)
    probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
    zero_state = np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=1) #positive energy eigenvector splitted in components
    spin_values.append(mean_spin_xy(np.expand_dims(zero_state[:L_x, :]/np.linalg.norm(zero_state[:L_x, :]), axis=1)))

for i in index:
    for j in range(len(phi_values)):
        for l in range(len(k_y_values)):
            edge_states[l, j, 0:4, i] = eigenvectors_k_phi[l, j, 4*(L_x-8):4*(L_x-7), i]
            edge_states[l, j, 4:8, i] = eigenvectors_k_phi[l, j, 4*(L_x+7):4*(L_x+8), i]
            edge_states[l, j, 8:12, i] = eigenvectors_k_phi[l, j, 4*7:4*8, i]
            edge_states[l, j, 12:16, i] = eigenvectors_k_phi[l, j, 4*(2*L_x-8):4*(2*L_x-7), i]

    # zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components
# for i in index:
#     destruction_up, destruction_down, creation_down, creation_up = get_components(eigenvectors[:,i], L_x, L_y)
#     probability_density.append((np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)/(np.linalg.norm(np.abs(destruction_up)**2 + np.abs(destruction_down)**2 + np.abs(creation_down)**2 + np.abs(creation_up)**2)))
#     zero_state.append(np.stack((destruction_up, destruction_down, creation_down, creation_up), axis=2)) #positive energy eigenvector splitted in components

index = 4

fig, ax = plt.subplots()
ax.plot(probability_density[index]) 
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

#%%
from functions import mean_spin, mean_spin_xy

# print(mean_spin(np.reshape(eigenvectors_sparse[4*(L_x-8):4*(L_x-7), index]/np.linalg.norm(eigenvectors_sparse[4*(L_x-8):4*(L_x-7), index]), (4,1))))
# print(mean_spin(np.reshape(eigenvectors_sparse[4*(L_x+7):4*(L_x+8), index]/np.linalg.norm(eigenvectors_sparse[4*(L_x+7):4*(L_x+8), index]), (4,1))))

print("ky<0 left")
print(mean_spin(np.reshape(edge_states[0, 0, 0:4, 0], (4,1))))  # ky<0 left
print("\n")

print("ky<0 right")
print(mean_spin(np.reshape(edge_states[0, 0, 4:8, 0], (4,1))))   # ky>0 right
print("\n")

print("ky=0 left")
print(mean_spin(np.reshape(edge_states[1, 0, 0:4, 2], (4,1))))  # ky=0 left
print(mean_spin(np.reshape(edge_states[1, 0, 0:4, 3], (4,1))))  # ky=0 left
print("\n")

print("ky=0 right")
print(mean_spin(np.reshape(edge_states[1, 0, 4:8, 2], (4,1))))   # ky=0 right
print(mean_spin(np.reshape(edge_states[1, 0, 4:8, 3], (4,1))))   # ky=0 right
print("\n")

print("ky>0 left")
print(mean_spin(np.reshape(edge_states[2, 0, 0:4, 0], (4,1))))  # ky>0 left
print("\n")

print("ky>0 right")
print(mean_spin(np.reshape(edge_states[2, 0, 4:8, 0], (4,1))))   # ky<0 right
print("\n")

# print("ky=0 edge left outside junction")
# print(mean_spin(np.reshape(edge_states[1, 0, 8:12, 2], (4,1))))  # ky=0 left left
# print(mean_spin(np.reshape(edge_states[1, 0, 8:12, 3], (4,1))))  # ky=0 left left
# print("\n")

# print("ky=0 edge right outside junction")
# print(mean_spin(np.reshape(edge_states[1, 0, 12:16, 0], (4,1))))  # ky=0 right right
# print(mean_spin(np.reshape(edge_states[1, 0, 12:16, 1], (4,1))))  # ky=0 right right

print("\n")
