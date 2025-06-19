#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:23:10 2023

@author: gabriel
"""
import numpy as np
import scipy
from pauli_matrices import tau_z, sigma_0, tau_0
from hamiltonian import Hamiltonian

class Junction(Hamiltonian):
    r"""
    Superconductors 1 and 2 should not be periodic.
    
    .. math ::
        
        H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)
              \tau_z\sigma_0
              +isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    def __init__(self, Superconductor_1, Superconductor_2, t_J, phi):
        self.t_J = t_J
        self.phi = phi
        self.L_x = Superconductor_1.L_x + Superconductor_2.L_x
        self.L_y = Superconductor_1.L_y        #check
        self.matrix = self._get_matrix(Superconductor_1, Superconductor_2)
    def _get_matrix(self, Superconductor_1, Superconductor_2):
        S_1 = Superconductor_1._get_matrix()
        S_2 = Superconductor_2._get_matrix()
        block_diagonal = scipy.sparse.bmat([[S_1, None], [None, S_2]])
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y, 4*self.L_x*self.L_y),
                                    dtype=complex)
        for j in range(1, self.L_y+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(Superconductor_1.L_x, j, alpha),
                      self._index(Superconductor_1.L_x + 1, j, beta)] =\
                        self._hopping_junction(j-1)[alpha, beta]
        return block_diagonal + M + M.conj().T
    def _hopping_junction(self, j):
        return self.t_J/2*( np.cos(self.phi[j]/2)*np.kron(tau_z, sigma_0)+
                          1j*np.sin(self.phi[j]/2)*np.kron(tau_0, sigma_0) )

class Junction_in_y(Hamiltonian):
    r"""
    Superconductors 1 and 2 should not be periodic.
    
    .. math ::
        
        H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)
              \tau_z\sigma_0
              +isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    def __init__(self, Superconductor_1, Superconductor_2, t_J, phi):
        self.t_J = t_J
        self.phi = phi
        self.L_x = Superconductor_1.L_x
        self.L_y = Superconductor_1.L_y + Superconductor_2.L_y      #check
        self.matrix = self._get_matrix(Superconductor_1, Superconductor_2)
    def _get_matrix(self, Superconductor_1, Superconductor_2):
        S_1 = Superconductor_1._get_matrix()
        S_2 = Superconductor_2._get_matrix()
        block_diagonal = scipy.sparse.bmat([[S_1, None], [None, S_2]])
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y, 4*self.L_x*self.L_y),
                                    dtype=complex)
        for i in range(1, self.L_x+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(i, Superconductor_1.L_y, alpha),
                      self._index(i, Superconductor_1.L_y + 1, beta)] =\
                        self._hopping_junction(i-1)[alpha, beta]
        return block_diagonal + M + M.conj().T
    def _hopping_junction(self, i):
        return self.t_J/2*( np.cos(self.phi[i]/2)*np.kron(tau_z, sigma_0)+
                          1j*np.sin(self.phi[i]/2)*np.kron(tau_0, sigma_0) )

class JunctionWithQD(Hamiltonian):
    r"""
    Superconductors 1 and 2 should not be periodic.
    
    .. math ::
        
        H_J = t_J/2\sum_m^{L_y}[\vec{c}_{L_x-1,m}^\dagger(cos(\phi/2)
              \tau_z\sigma_0
              +isin(\phi/2)\tau_0\sigma_0)\vec{c}_{L_x,m}+H.c.]
    """
    def __init__(self, Superconductor_1, Superconductor_2, t_J, phi):
        self.t_J = t_J
        self.phi = phi
        self.L_x = Superconductor_1.L_x + Superconductor_2.L_x + 1
        self.L_y = Superconductor_1.L_y        #check
        self.matrix = self._get_matrix(Superconductor_1, Superconductor_2)
    def _get_matrix(self, Superconductor_1, Superconductor_2):
        S_1 = Superconductor_1._get_matrix()
        S_2 = Superconductor_2._get_matrix()
        block_diagonal = scipy.sparse.bmat([[S_1, None, None],
                                            [None, np.eye(4) * np.diag(Superconductor_1.onsite), None],
                                            [None, None, S_2]])
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y, 4*self.L_x*self.L_y),
                                    dtype=complex)
        for j in range(1, self.L_y+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(Superconductor_1.L_x, j, alpha),
                      self._index(Superconductor_1.L_x + 1, j, beta)] =\
                        self._hopping_junction(j-1)[alpha, beta]
                    M[self._index(Superconductor_1.L_x + 1, j, alpha),
                      self._index(Superconductor_1.L_x + 2, j, beta)] =\
                        self._hopping_junction(j-1)[alpha, beta]
        return block_diagonal + M + M.conj().T
    def _hopping_junction(self, j):
        return self.t_J/2*( np.cos(self.phi[j]/4)*np.kron(tau_z, sigma_0)+
                          1j*np.sin(self.phi[j]/4)*np.kron(tau_0, sigma_0) )

class PeriodicJunction(Junction):
    """Superconductors 1 and 2 should not be periodic."""
    def __init__(self, Superconductor_1, Superconductor_2, t_J, phi):
        self.t_J = t_J
        self.phi = phi
        self.L_x = Superconductor_1.L_x + Superconductor_2.L_x
        self.L_y = Superconductor_1.L_y
        self.matrix = self._get_matrix(Superconductor_1, Superconductor_2) +\
                            self._get_matrix_periodic_in_y(Superconductor_1,
                                                           Superconductor_2) 
    def _get_matrix_periodic_in_y(self, Superconductor_1, Superconductor_2):
        """Returns the part of the tight binding matrix which connects the first
        and  the last site in the y direction."""
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y,
                                     4*self.L_x*self.L_y),
                                    dtype=complex)
        #hopping_y, periodic boundary conditions in Superconductor 1
        for i in range(1, Superconductor_1.L_x+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(i, self.L_y, alpha),
                      self._index(i, 1, beta)] = \
                        Superconductor_1.hopping_y[alpha, beta]
        #hopping_y, periodic boundary conditions in Superconductor 2                        
        for i in range(Superconductor_1.L_x+1, self.L_x):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(i, self.L_y, alpha),
                      self._index(i, 1, beta)] = \
                        Superconductor_2.hopping_y[alpha, beta]
        return M + M.conj().T

class PeriodicJunctionInX(Junction):
    """Superconductors 1 and 2 should not be periodic."""
    def __init__(self, Superconductor_1, Superconductor_2, t_J, phi):
        self.t_J = t_J
        self.phi = phi
        self.L_x = Superconductor_1.L_x + Superconductor_2.L_x
        self.L_y = Superconductor_1.L_y
        self.matrix = self._get_matrix(Superconductor_1, Superconductor_2) +\
                            self._get_matrix_periodic_in_x(Superconductor_1,
                                                           Superconductor_2) 
    def _get_matrix_periodic_in_x(self, Superconductor_1, Superconductor_2):
        """Returns the part of the tight binding matrix which connects the first
        and  the last site in the x direction."""
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y,
                                     4*self.L_x*self.L_y),
                                    dtype=complex)
        #hopping_x, periodic boundary conditions in Superconductor 1
        for j in range(1, Superconductor_1.L_y+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(self.L_x, j, alpha),
                      self._index(1, j, beta)] = \
                        Superconductor_1.hopping_x[alpha, beta]
        return M + M.conj().T
    
class PeriodicJunctionInXWithQD(JunctionWithQD, PeriodicJunctionInX):
    """Superconductors 1 and 2 should not be periodic."""
    def __init__(self, Superconductor_1, Superconductor_2, t_J, phi):
        self.t_J = t_J
        self.phi = phi
        self.L_x = Superconductor_1.L_x + Superconductor_2.L_x + 1
        self.L_y = Superconductor_1.L_y
        self.matrix = self._get_matrix(Superconductor_1, Superconductor_2) +\
                            self._get_matrix_periodic_in_x(Superconductor_1,
                                                           Superconductor_2)
        
class PeriodicJunctionInXAndY(PeriodicJunction):
    """Superconductors 1 and 2 should not be periodic."""
    def __init__(self, Superconductor_1, Superconductor_2, t_J, phi):
        super().__init__(Superconductor_1, Superconductor_2, t_J, phi)
        self.matrix = self._get_matrix(Superconductor_1, Superconductor_2) +\
                            self._get_matrix_periodic_in_y(Superconductor_1,
                                                           Superconductor_2)+\
                            self._get_matrix_periodic_in_x(Superconductor_1,
                                                           Superconductor_2)    
    def _get_matrix_periodic_in_x(self, Superconductor_1, Superconductor_2):
        """Returns the part of the tight binding matrix which connects the first
        and  the last site in the x direction."""
        M = scipy.sparse.lil_matrix((4*self.L_x*self.L_y,
                                     4*self.L_x*self.L_y),
                                    dtype=complex)
        #hopping_y, periodic boundary conditions in Superconductor 1
        for j in range(1, Superconductor_1.L_y+1):
            for alpha in range(4):
                for beta in range(4):
                    M[self._index(self.L_x, j, alpha),
                      self._index(1, j, beta)] = \
                        Superconductor_1.hopping_x[alpha, beta]
        return M + M.conj().T
