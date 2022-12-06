import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, csr_matrix


class Transmon():
    """
    Class to create a transmon with given parameters. Can be generalized through the PUK

    We create from a dictionairy with the following:
    n_cutoff        - To determine size of operators
    e               - constant

    EJ              - Josephson energy
    EJ_EC_ratio     - Ratio to determine capacitance of the circuit
    gamma           - The EJ2 / EJ ratio    
    """

    def __init__(self, define_dict):
        # Can be created either in charge or flux basis. 
        self.n_cutoff = define_dict["n_cutoff"]
        self.n        = self.n_cutoff * 2 + 1 # Dimensions of hilbert space
        self.e        = define_dict["e"]

        # Hamiltonian parameters (from fabrication)
        self.EJ       = define_dict["EJ"]
        self.EC       = self.EJ / define_dict["EJ_EC_ratio"]
        
        # Define second Josephson junction
        if "gamma" in define_dict.keys():
            self.gamma      = define_dict["gamma"]
            self.EJ2        = self.EJ * self.gamma
        else:
            self.gamma      = None
            self.EJ2        = None

    def charge(self):
        return np.arange(-self.n_cutoff, self.n_cutoff + 1, 1)

    def Hamiltonian(self, charge_offset = 0, external_flux = 0):
        # Get kinetic and potential term
        kinetic     = self.kinetic(charge_offset=charge_offset, external_flux=external_flux)
        potential   = self.V(charge_offset=charge_offset, external_flux=external_flux)

        diag_potential = diags(potential)

        return diag_potential + kinetic

    def kinetic(self, charge_offset = 0, external_flux = 0):
        # Combine circuit to get the flux terms in (22) from A Quantum engineers guide
        if self.gamma:
            EJ_sum          = self.EJ + self.EJ2
            d               = (self.gamma - 1) / (self.gamma + 1)
            coefficient     = - EJ_sum * np.sqrt(np.cos(external_flux) ** 2 + d ** 2 * np.sin(external_flux) ** 2) 
        else:
            coefficient     = - self.EJ
        
        return coefficient * self.create_cos_matrix()
        
    def V(self, charge_offset = 0, external_flux = 0):
        # Combine circuit to get (22) from A Quantum Engineers Guide ... including charge
        n_diag      = np.arange(- self.n_cutoff, self.n_cutoff + 1, 1)
        V           = 4 * self.EC * (n_diag + charge_offset) ** 2
        return V

    def V_in_flux_basis(self, charge_offset = 0, external_flux = 0):
        raise NotImplementedError()

    def q_matrix(self):
        # Matrix to get the charge matrix
        diagonal    = np.arange(- self.n_cutoff, self.n_cutoff + 1, 1)
        q_matrix    = diags(diagonal)
        return 2 * self.e * q_matrix

    def n_matrix(self):
        # Matrix to get jumps. Equal to n_matrix / 2 e
        diagonal    = np.arange(- self.n_cutoff, self.n_cutoff + 1, 1)
        q_matrix    = diags(diagonal)
        return q_matrix

    def exp_i_flux(self, cyclic = True):
        # Exponent of the flux. We implement it as sum_n |n><n+1|

        n = self.n_cutoff * 2 + 1
        off_diag = np.ones(n - 1)
        off_diag_sparse = diags(off_diag, offsets = 1)
        cos_matrix = off_diag_sparse

        if cyclic:
            cyclic_component = csr_matrix(([1], ([n-1], [0])), shape = (n, n))
        
        return (cos_matrix + cyclic_component)

    def create_cos_matrix(self, cyclic = True):
        # Combine exp_i_flux to get cos = exp(i []) + exp(- i [])
        exp_flux    = self.exp_i_flux(cyclic = cyclic)
        cos_matrix  = (exp_flux + exp_flux.getH()) / 2
        return cos_matrix
        
    def fourier_transform_matrix(self):
        n       = self.n_cutoff * 2 + 1
        qs      = np.arange(-self.n_cutoff, self.n_cutoff + 1, 1)
        phis    = np.linspace(- np.pi, np.pi, n)

        return 1 / np.sqrt(n) * np.exp(1j * np.outer(qs, phis))
    

from scipy.stats import norm

class PulseGenerator():
    """
    Define a class to create different pulses 
    """
    def __init__(self, omega, magnitude = np.pi, envelope = "gaussian", arguments = None):

        # Get the envelope of the function 
        self.s  = self.get_envelope(envelope, arguments)

        # Create pulse
        self.A     = magnitude
        self.omega = omega
        self.pulse = self.get_pulse()


    # Put together
    def total_func(self):
        def f(t, phase = 0):
            return self.A * self.pulse(t, phase = phase) * self.s(t)
        return f

    def get_pulse(self):
        def cosine_term(t, phase = 0):
            return np.cos(self.omega * t + phase)
        return cosine_term


    def get_envelope(self, envelope, arguments = None):
        if envelope in ["gaussian", "norm"]:
            T       = arguments["T"]
            width   = arguments["width"] 

            s       = lambda t: norm.pdf(t, loc = T, scale = width)

            return s 
            
            




if __name__ == "__main__":
    creation_dict = {
        "n_cutoff":         10,
        "e":                1e-12,
        "EJ":               1,
        "EJ_EC_ratio":      50, 
        "gamma":            2.5
    }

    circuit = Transmon(creation_dict)
    plt.close("all")
    plt.imshow(circuit.Hamiltonian().todense())
    plt.show()

