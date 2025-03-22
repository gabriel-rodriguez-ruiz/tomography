# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:52:18 2024

@author: Gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductorKY
from functions import get_components       
from junction import Junction, PeriodicJunction
import scipy
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": False})

L_x = 300
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
t_J = t/2    #t/2#t/5
phi_values = np.linspace(0, 2*np.pi, 100)    #240
k_y_values = [2*np.pi/100]  #np.array([-2*np.pi/100, 0, 2*np.pi/100]) #np.linspace(0, 2*np.pi, 20)   #np.array([-2*np.pi/100, 0, 2*np.pi/100]) #np.linspace(0, 2*np.pi, 200)  #200
antiparallel = False

params = {"L_x":L_x, "t":t, "t_J":t_J,
          "Delta_0":Delta_0,
          "Delta_1":Delta_1,
          "mu":mu, "phi_values":phi_values,
          "k_y_values": k_y_values,
          "B": B, "phi_angle": phi_angle,
          "theta": theta, "antiparallel": antiparallel
          }

eigenvalues_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x))
eigenvectors_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x, 2*4*L_x), dtype=complex)
for i, k_y in enumerate(k_y_values):
    print(k_y)
    for j, phi in enumerate(phi_values):
        phi = np.array([phi])   #array of length 1
        S_ZKMB = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x, B_y, B_z)
        S_ZKMB2 = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x=(1-2*antiparallel)*B_x, B_y=(1-2*antiparallel)*B_y, B_z=(1-2*antiparallel)*B_z)
        J = Junction(S_ZKMB, S_ZKMB2, t_J, phi)
        eigenvalues, eigenvectors = np.linalg.eigh(J.matrix.toarray())
        eigenvalues_k_phi[i, j, :] = eigenvalues
        eigenvectors_k_phi[i, j, :, :] = eigenvectors

def get_coefficients(eigenvectors_k_phi):
    N_k, N_phi, N, M = np.shape(eigenvectors_k_phi)
    A_minus_k_phi_nu_1_R_up = -eigenvectors_k_phi[:, :, 8*(L_x-1)//2+3, :].conj()
    A_minus_k_phi_nu_1_R_down = eigenvectors_k_phi[:, :,  8*(L_x-1)//2+2, :].conj()
    A_minus_k_phi_nu_2_L_up = -eigenvectors_k_phi[:, :, 8*L_x//2+3, :].conj()
    A_minus_k_phi_nu_2_L_down = eigenvectors_k_phi[:, :, 8*L_x//2+2, :].conj()
    return A_minus_k_phi_nu_1_R_up, A_minus_k_phi_nu_1_R_down, A_minus_k_phi_nu_2_L_up, A_minus_k_phi_nu_2_L_down


#%%

def get_Josephson_current(eigenvalues_k_phi, eigenvectors_k_phi):
    N_k, N_phi, N, M = np.shape(eigenvectors_k_phi)
    J_k_phi = np.zeros((N_k, N_phi), dtype=complex)
    A_minus_k_phi_nu_1_R_up, A_minus_k_phi_nu_1_R_down, A_minus_k_phi_nu_2_L_up, A_minus_k_phi_nu_2_L_down = get_coefficients(eigenvectors_k_phi)
    sumand = np.conj(A_minus_k_phi_nu_1_R_up) * A_minus_k_phi_nu_2_L_up + np.conj(A_minus_k_phi_nu_1_R_down) * A_minus_k_phi_nu_2_L_down
    positive_energy_sumand = np.where(eigenvalues_k_phi>0, sumand, np.zeros_like(sumand))
    for i in range(N_k):
        for j in range(N_phi):
            J_k_phi[i, j] = -np.imag(np.exp(1j*phi_values[j]/2) * np.sum(positive_energy_sumand[i, j, :]))
    return J_k_phi


#%% Josephson current

Josephson_current_k = get_Josephson_current(eigenvalues_k_phi, eigenvectors_k_phi)
Josephson_current = np.sum(Josephson_current_k, axis=0)

J_0 = np.max(Josephson_current) 
fig, ax = plt.subplots()
ax.plot(phi_values/(2*np.pi), Josephson_current/J_0)
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J(\phi)/J_0$")
ax.set_title("Josephson current")

fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")
ax.set_title("Josephson current for given k\n"+
             r"$\theta=$" + f"{np.round(theta, 2)}"+
             r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
             r"; $B=$" + f"{B}")

for i, k in enumerate(k_y_values):
    ax.scatter(phi_values/(2*np.pi), Josephson_current_k[i,:],
               marker=".",
               label=r"$k_y=$" + f"{np.round(k_y_values[i], 2)}")

ax.legend(fontsize= "xx-small")
plt.show()

#%% Save 

np.savez(f"Data/Josephson_current_theta_{np.round(theta,3)}_phi_angle_{np.round(phi_angle, 3)}_phi_values_{len(phi_values)}_k_y_values_{len(k_y_values)}_tJ_{np.round(t_J, 3)}_antiparallel_{antiparallel}", Josephson_current=Josephson_current, J_0=J_0,
         Josephson_current_k=Josephson_current_k,
        params=params, k_y_values=k_y_values, phi_values=phi_values)