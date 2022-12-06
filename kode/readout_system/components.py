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
    


class Resonator():
    """
    This class creates an LC circuit, that we can couple to the Transmon
    """

    def __init__(self, omega, n_cutoff = 4):
        # Define parameters
        self.omega      = omega
        self.n_cutoff   = n_cutoff

    def Hamiltonian(self):
        return self.omega * self.a_dagger() @ self.a() 

    def a(self): 
        n           = self.occupation()
        off_diag    = np.sqrt(n[:-1])
        return diags(off_diag, +1)

    def a_dagger(self):
        return np.conjugate(self.a()).T

    def occupation(self):
        occupation = np.arange(1, self.n_cutoff + 1)
        return occupation


from scipy.stats import norm
class GaussianPulseGenerator():

    def __init__(self, T, width, omega, phase = 0, drag = False):
        # Load params
        self.T      = T
        self.width  = width
        self.omega  = omega

        # Find I and Q
        self.I      = np.cos(phase)
        self.Q      = np.sin(phase)

        self.drag   = drag

        # Envelopes:
        self.epsilon_x, self.epsilon_y = self.envelopes()

        # Pulses
        self.I_pulse, self.Q_pulse = self.pulses()

        


    def pulses(self):
        I_pulse = lambda t: np.cos(self.omega * t)
        Q_pulse = lambda t: np.sin(self.omega * t)
        return I_pulse, Q_pulse


    def envelopes(self):
        
        def epsilon_x(t):
            A = 1 / np.sqrt(2 * np.pi) / self.width
            return A * np.exp((t - self.T) ** 2 / self.width ** 2 / 2)
        
        if self.drag:
            def epsilon_y(t):
                A = 1 / np.sqrt(2 * np.pi) / self.width
                B = - 1 / (2 * self.width ** 2) 
                return A * B * (2 * t) * np.exp((t - self.T) ** 2 / self.width ** 2 / 2)
        else:
            epsilon_y = epsilon_x

        return epsilon_x, epsilon_y








if __name__ == "__main__":
    res = Resonator(omega = 1)
    print(res.Hamiltonian().todense())
