# Define the operators
from qutip import charge, Qobj
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import numpy as np

class Transmon():
    """
    Transmon class. Can possibly be generalized to account for flux_basis and shunted with a junction array.
    """
    def __init__(self, EC, EJ, basis = "charge", n_cutoff = 10):
        # Setup the meta parameters
        self.basis = basis
        self.n_cutoff = n_cutoff
        self.n = self.n_cutoff * 2 + 1

        # Energy of the circuit elements
        self.EC = EC
        self.EJ = EJ

        if self.basis != "charge":
            raise NotImplementedError()
        
        # Get operators
        self.charge, self.cos_flux = self.operators()

        # Define the Hamiltonian
        self.H = self.Hamiltonian(charge_offset = 0, external_flux = 0)

    def lowest_k_eigenstates(self, k = 2):
        H = self.H.data.todense()
        charge_op = self.charge.data.todense()

        vals, vecs = eigsh(H, k = k, which = "SA")

        H_reduced       = Qobj(np.diag(vals))
        charge_overlap  = Qobj(vecs.T.conj() @ charge_op @ vecs)

        return H_reduced, charge_overlap
    
    def Hamiltonian(self, charge_offset = 0, external_flux = 0):
        # Get kinetic and potential term
        if external_flux != 0:
            raise NotImplementedError()
        
        kinetic     = 4 * self.EC * (self.charge - charge_offset) ** 2
        potential   = - self.EJ * self.cos_flux

        return kinetic + potential 


    def operators(self):
        # Define the operators in the flux basis
        charge_op = charge(self.n_cutoff)

        # Define the flux operators
        exp_i_flux = diags(np.ones(self.n - 1), offsets = -1)
        cos_flux   = Qobj((exp_i_flux + exp_i_flux.getH()) / 2)
        
        return charge_op, cos_flux 