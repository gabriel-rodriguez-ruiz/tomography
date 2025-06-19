# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 08:52:18 2024

@author: Gabriel
"""

import numpy as np
from ZKMBsuperconductor import ZKMBSuperconductorKY, ZKMBSuperconductorKX
from functions import get_components       
from junction import Junction, Junction_in_y, JunctionWithQD
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
phi_angle = np.pi/2 #np.pi/8 #np.pi/16#np.pi/2
theta = np.pi/2 #np.pi/2   #np.pi/2
B = 2*Delta_0   #2*Delta_0
B_x = B * np.sin(theta) * np.cos(phi_angle)
B_y = B * np.sin(theta) * np.sin(phi_angle)
B_z = B * np.cos(theta)
mu = -4*t#-2*t
t_J = t/2    #t/2#t/5
# phi_values = np.sort(np.append(np.linspace(0, 2*np.pi, 50), np.pi))    #240
phi_values = np.linspace(0, 2*np.pi, 100)    #240
k_y_values = [0]  #np.array([-0.01, 0, 0.01])   #np.linspace(0, 2*np.pi, 10)   #np.array([-2*np.pi/100, 0, 2*np.pi/100]) #np.linspace(0, 2*np.pi, 200)  #200
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

# eigenvalues_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x))
# eigenvectors_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x, 2*4*L_x), dtype=complex)
eigenvalues_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x+4))             # Lx > L_x+4
eigenvectors_k_phi = np.zeros((len(k_y_values), len(phi_values), 2*4*L_x+4, 2*4*L_x+4), dtype=complex)

for i, k_y in enumerate(k_y_values):
    print(k_y)
    for j, phi in enumerate(phi_values):
        phi = np.array([phi])   #array of length 1
        S_ZKMB = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x, B_y, B_z)
        S_ZKMB2 = ZKMBSuperconductorKY(k_y, L_x, t, mu, Delta_0, Delta_1,
                                      Lambda, B_x=(1-2*antiparallel)*B_x, B_y=(1-2*antiparallel)*B_y, B_z=(1-2*antiparallel)*B_z)
        # J = Junction(S_ZKMB, S_ZKMB2, t_J, phi)
        J = JunctionWithQD(S_ZKMB, S_ZKMB2, t_J, phi)
        eigenvalues, eigenvectors = np.linalg.eigh(J.matrix.toarray())
        eigenvalues_k_phi[i, j, :] = eigenvalues
        eigenvectors_k_phi[i, j, :, :] = eigenvectors

# eigenvalues_k_phi = np.zeros((len(k_x_values), len(phi_values), 2*4*L_y))
# eigenvectors_k_phi = np.zeros((len(k_x_values), len(phi_values), 2*4*L_y, 2*4*L_y), dtype=complex)
# for i, k_x in enumerate(k_x_values):
#     print(k_x)
#     for j, phi in enumerate(phi_values):
#         phi = np.array([phi])   #array of length 1
#         S_ZKMB = ZKMBSuperconductorKX(k_x, L_y, t, mu, Delta_0, Delta_1,
#                                       Lambda, B_x, B_y, B_z)
#         S_ZKMB2 = ZKMBSuperconductorKX(k_x, L_y, t, mu, Delta_0, Delta_1,
#                                       Lambda, B_x=(1-2*antiparallel)*B_x, B_y=(1-2*antiparallel)*B_y, B_z=(1-2*antiparallel)*B_z)
#         # J = Junction(S_ZKMB, S_ZKMB2, t_J, phi)
#         J = Junction_in_y(S_ZKMB, S_ZKMB2, t_J, phi)
#         eigenvalues, eigenvectors = np.linalg.eigh(J.matrix.toarray())
#         eigenvalues_k_phi[i, j, :] = eigenvalues
#         eigenvectors_k_phi[i, j, :, :] = eigenvectors

#%%
total_energy_k = np.zeros((len(k_y_values), len(phi_values)))

for i in range(len(k_y_values)):
    E_positive = np.where(eigenvalues_k_phi[i, :, :] > 0, eigenvalues_k_phi, np.zeros_like(eigenvalues_k_phi))
    total_energy_k[i, :] = np.sum(E_positive, axis=2)

dphi = np.diff(phi_values)
Josephson_current_k_eigenvalues = np.gradient(-total_energy_k, dphi[0], axis=1)

fig, ax = plt.subplots()
ax.plot(phi_values, total_energy_k[0, :], "ok", markersize=0.5)
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.scatter(phi_values/(2*np.pi), Josephson_current_k_eigenvalues[i,:],
           marker=".",
           label=r"$|k_y|=$" + f"{np.abs(np.round(k_y_values[i], 3))}")
ax.legend()
plt.show()
#%% Total energy

total_energy_k = np.zeros((len(k_y_values)//2 + 1, len(phi_values)))

for i in range(len(k_y_values)//2 + 1):
    E_phi_k_minus_k = (eigenvalues_k_phi[i, :, :] + eigenvalues_k_phi[len(k_y_values)-(i+1), :, :]) / 2
    E_positive = np.where(E_phi_k_minus_k > 0, E_phi_k_minus_k, np.zeros_like(E_phi_k_minus_k))
    total_energy_k[i] = np.sum(E_positive, axis=1)

fig, ax = plt.subplots()
ax.plot(phi_values, total_energy_k[0, :], "ok", markersize=0.5)
# ax.plot(phi_values, total_energy_k[1, :], "or", label=f"{k_y_values[1]}", markersize=0.5)
ax.legend()
plt.show()
#%% Josephson current

dphi = np.diff(phi_values)
# Josephson_current_k_eigenvalues = np.diff(-total_energy_k) / dphi
Josephson_current_k_eigenvalues = np.gradient(-total_energy_k, dphi[0], axis=1)
Josephson_current_eigenvalues = np.sum(Josephson_current_k_eigenvalues, axis=0)

fig, ax = plt.subplots()
ax.set_xlabel(r"$\phi/(2\pi)$")
ax.set_ylabel(r"$J_k(\phi)$")   
ax.set_title("Josephson current for given k\n"+
             r"$\theta=$" + f"{np.round(theta, 2)}"+
             r"; $\varphi=$" + f"{np.round(phi_angle, 2)}"
             r"; $B=$" + f"{B}")

for i in range(len(k_y_values)//2 + 1):
    # ax.scatter(phi_values[:-1]/(2*np.pi), Josephson_current_k_eigenvalues[i,:],
    #            marker=".",
    #            label=r"$|k_y|=$" + f"{np.abs(np.round(k_y_values[i], 3))}")
    ax.scatter(phi_values/(2*np.pi), Josephson_current_k_eigenvalues[i,:],
               marker=".",
               label=r"$|k_y|=$" + f"{np.abs(np.round(k_y_values[i], 3))}")
ax.legend(fontsize= "xx-small")
plt.show()

#%% Get coefficients

def get_coefficients(eigenvectors_k_phi):
    N_k, N_phi, N, M = np.shape(eigenvectors_k_phi)
    A_minus_k_phi_nu_1_R_up = -eigenvectors_k_phi[:, :, 8*(L_x-1)//2+3].conj()
    A_minus_k_phi_nu_1_R_down = eigenvectors_k_phi[:, :, 8*(L_x-1)//2+2].conj()
    A_minus_k_phi_nu_2_L_up = -eigenvectors_k_phi[:, :, 8*L_x//2+3].conj()
    A_minus_k_phi_nu_2_L_down = eigenvectors_k_phi[:, :, 8*L_x//2+2].conj()
    A_plus_k_phi_nu_1_R_up = eigenvectors_k_phi[:, :, 8*(L_x-1)//2, :].conj()
    A_plus_k_phi_nu_1_R_down = eigenvectors_k_phi[:, :,  8*(L_x-1)//2+1, :].conj()
    A_plus_k_phi_nu_2_L_up = -eigenvectors_k_phi[:, :, 8*L_x//2, :].conj()
    A_plus_k_phi_nu_2_L_down = eigenvectors_k_phi[:, :, 8*L_x//2+1, :].conj()
    return A_minus_k_phi_nu_1_R_up, A_minus_k_phi_nu_1_R_down, A_minus_k_phi_nu_2_L_up, A_minus_k_phi_nu_2_L_down,\
            A_plus_k_phi_nu_1_R_up, A_plus_k_phi_nu_1_R_down, A_plus_k_phi_nu_2_L_up, A_plus_k_phi_nu_2_L_down

# def get_coefficients(eigenvectors_k_phi):
#     N_k, N_phi, N, M = np.shape(eigenvectors_k_phi)
#     A_minus_k_phi_nu_1_R_up = -eigenvectors_k_phi[:, :, 8*(L_x-1)//2+3, :]
#     A_minus_k_phi_nu_1_R_down = eigenvectors_k_phi[:, :,  8*(L_x-1)//2+2, :]
#     A_minus_k_phi_nu_2_L_up = -eigenvectors_k_phi[:, :, 8*L_x//2+3, :] 
#     A_minus_k_phi_nu_2_L_down = eigenvectors_k_phi[:, :, 8*L_x//2+2, :]
#     return A_minus_k_phi_nu_1_R_up, A_minus_k_phi_nu_1_R_down, A_minus_k_phi_nu_2_L_up, A_minus_k_phi_nu_2_L_down

#%% Plot coefficients
A_minus_k_phi_nu_1_R_up = -eigenvectors_k_phi[:, :, 8*(L_x-1)//2+3, :].conj()
A_minus_k_phi_nu_1_R_down = eigenvectors_k_phi[:, :,  8*(L_x-1)//2+2, :].conj()
A_minus_k_phi_nu_2_L_up = -eigenvectors_k_phi[:, :, 8*L_x//2+3, :].conj()
A_minus_k_phi_nu_2_L_down = eigenvectors_k_phi[:, :, 8*L_x//2+2, :].conj()

A_plus_k_phi_nu_1_R_up = eigenvectors_k_phi[:, :, 8*(L_x-1)//2, :].conj()
A_plus_k_phi_nu_1_R_down = eigenvectors_k_phi[:, :,  8*(L_x-1)//2+1, :].conj()
A_plus_k_phi_nu_2_L_up = -eigenvectors_k_phi[:, :, 8*L_x//2, :].conj()
A_plus_k_phi_nu_2_L_down = eigenvectors_k_phi[:, :, 8*L_x//2+1, :].conj()

positive_energy_sumand = np.zeros((len(k_y_values)//2 + 1, len(phi_values), 8*L_x))

for i in range(len(k_y_values)//2 + 1):
    A_minus_k_phi_nu_1_R_up = -( eigenvectors_k_phi[i, :, 8*(L_x-1)//2+3, :] + eigenvectors_k_phi[len(k_y_values)-(i+1), :, 8*(L_x-1)//2+3, :]).conj() / 2
    A_minus_k_phi_nu_1_R_down = ( eigenvectors_k_phi[i, :,  8*(L_x-1)//2+2, :] + eigenvectors_k_phi[len(k_y_values)-(i+1), :,  8*(L_x-1)//2+2, :]).conj() / 2
    A_minus_k_phi_nu_2_L_up = -( eigenvectors_k_phi[i, :, 8*L_x//2+3, :] + eigenvectors_k_phi[len(k_y_values)-(i+1), :, 8*L_x//2+3, :] ).conj() / 2
    A_minus_k_phi_nu_2_L_down = ( eigenvectors_k_phi[i, :, 8*L_x//2+2, :] + eigenvectors_k_phi[len(k_y_values)-(i+1), :, 8*L_x//2+2, :]).conj() / 2
    sumand = np.conj(A_minus_k_phi_nu_1_R_up) * A_minus_k_phi_nu_2_L_up + np.conj(A_minus_k_phi_nu_1_R_down) * A_minus_k_phi_nu_2_L_down
    positive_energy_sumand[i, :, :] = np.where((eigenvalues_k_phi[i, :, :] + eigenvalues_k_phi[len(k_y_values)-(i+1), :, :])/2>0, sumand, np.zeros_like(sumand))

N_k, N_phi, N, M = np.shape(eigenvectors_k_phi)
J_k_phi = np.zeros((len(k_y_values)//2 + 1, N_phi), dtype=complex)

for i in range(len(k_y_values)//2 + 1):
    for j in range(N_phi):
        J_k_phi[i, j] = -t_J*np.imag(np.exp(1j*phi_values[j]/2) * np.sum(positive_energy_sumand[i, j, :]))
# sumand = (np.conj(A_minus_k_phi_nu_1_R_up) * A_minus_k_phi_nu_2_L_up +
#           np.conj(A_minus_k_phi_nu_1_R_down) * A_minus_k_phi_nu_2_L_down +
#           np.conj(A_plus_k_phi_nu_1_R_up) * A_plus_k_phi_nu_2_L_up +
#           np.conj(A_plus_k_phi_nu_1_R_down) * A_plus_k_phi_nu_2_L_down)
# sumand = np.conj(A_plus_k_phi_nu_1_R_up) * A_plus_k_phi_nu_2_L_up + np.conj(A_plus_k_phi_nu_1_R_down) * A_plus_k_phi_nu_2_L_down

# positive_energy_sumand = np.where(eigenvalues_k_phi>0, sumand, np.zeros_like(sumand))

fig, ax = plt.subplots()
# ax.plot(phi_values, sumand[0, :, 8*(L_x)//2])
# ax.plot(phi_values, positive_energy_sumand[0, :, 4*L_x+1])
# ax.plot(phi_values, positive_energy_sumand[1, :, 4*L_x+1])
# ax.plot(phi_values, positive_energy_sumand[2, :, 4*L_x+1])
for i, k in enumerate(range(len(k_y_values)//2 + 1)):
    ax.scatter(phi_values/(2*np.pi), J_k_phi[i,:],
               marker=".",
               label=r"$k_y=$" + f"{np.round(k_y_values[i], 2)}")

#%%

def get_Josephson_current(eigenvalues_k_phi, eigenvectors_k_phi):
    N_k, N_phi, N, M = np.shape(eigenvectors_k_phi)
    J_k_phi = np.zeros((N_k, N_phi), dtype=complex)
    A_minus_k_phi_nu_1_R_up, A_minus_k_phi_nu_1_R_down, A_minus_k_phi_nu_2_L_up, A_minus_k_phi_nu_2_L_down,\
            A_plus_k_phi_nu_1_R_up, A_plus_k_phi_nu_1_R_down, A_plus_k_phi_nu_2_L_up, A_plus_k_phi_nu_2_L_down = get_coefficients(eigenvectors_k_phi)
    sumand = np.conj(A_minus_k_phi_nu_1_R_up) * A_minus_k_phi_nu_2_L_up + np.conj(A_minus_k_phi_nu_1_R_down) * A_minus_k_phi_nu_2_L_down
    # sumand = (np.conj(A_minus_k_phi_nu_1_R_up) * A_minus_k_phi_nu_2_L_up +
    #           np.conj(A_minus_k_phi_nu_1_R_down) * A_minus_k_phi_nu_2_L_down +
    #           np.conj(A_plus_k_phi_nu_1_R_up) * A_plus_k_phi_nu_2_L_up +
    #           np.conj(A_plus_k_phi_nu_1_R_down) * A_plus_k_phi_nu_2_L_down)
    # positive_energy_sumand = np.where(np.gradient(np.gradient(eigenvalues_k_phi, axis=0), axis=0)>0, sumand, np.zeros_like(sumand))
    positive_energy_sumand = np.where(eigenvalues_k_phi>0, sumand, np.zeros_like(sumand))
    # positive_energy_sumand = sumand
    for i in range(N_k):
        for j in range(N_phi):
            J_k_phi[i, j] = -t_J*np.imag(np.exp(1j*phi_values[j]/2) * np.sum(positive_energy_sumand[i, j, :]))
    return J_k_phi


#%% Josephson current

Josephson_current_k_eigenstates = get_Josephson_current(eigenvalues_k_phi, eigenvectors_k_phi)
Josephson_current_eigenstates = np.sum(Josephson_current_k_eigenstates, axis=0)

J_0 = np.max(Josephson_current_eigenstates) 
fig, ax = plt.subplots()
ax.plot(phi_values/(2*np.pi), Josephson_current_eigenstates/J_0)
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
    ax.scatter(phi_values/(2*np.pi), Josephson_current_k_eigenstates[i,:],
               marker=".",
               label=r"$k_y=$" + f"{np.round(k_y_values[i], 2)}")
    # ax.scatter(phi_values[:-1]/(2*np.pi), Josephson_current_k_eigenvalues[i,:],
    #            marker=".",
    #            label=r"$|k_y|=$" + f"{np.round(k_y_values[i], 2)}")
    ax.scatter(phi_values/(2*np.pi), Josephson_current_k_eigenvalues[i,:],
               marker=".",
               label=r"$|k_y|=$" + f"{np.round(k_y_values[i], 2)}")
ax.legend(fontsize= "xx-small")
plt.show()

#%% Save 

np.savez(f"Data/Josephson_current_theta_{np.round(theta,3)}_phi_angle_{np.round(phi_angle, 3)}_phi_values_{len(phi_values)}_k_y_values_{len(k_y_values)}_tJ_{np.round(t_J, 3)}_antiparallel_{antiparallel}", Josephson_current_eigenvalues=Josephson_current_eigenvalues, J_0=J_0,
         Josephson_current_k_eigenvalues=Josephson_current_k_eigenvalues, Josephson_current_k_eigenstates=Josephson_current_k_eigenstates,
        params=params, k_y_values=k_y_values, phi_values=phi_values)