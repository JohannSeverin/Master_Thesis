import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import eigsh



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

    def __init__(self, n_cutoff = 10, e = 1, EJ = 10, EJ_EC_ratio = 50, gamma = None):
        # Can be created either in charge or flux basis. 
        self.n_cutoff = n_cutoff
        self.n        = self.n_cutoff * 2 + 1 # Dimensions of hilbert space
        self.e        = e

        # Hamiltonian parameters (from fabrication)
        self.EJ       = EJ

        # Define second Josephson junction
        if gamma:
            self.gamma      = gamma
            self.EJ2        = self.EJ * self.gamma
            self.EC         = (self.EJ + self.EJ2) / EJ_EC_ratio
        else:
            self.gamma      = None
            self.EJ2        = None
            self.EC         = self.EJ / EJ_EC_ratio

    def eigen_basis(self, n = 2, charge_offset = 0, external_flux = 0):
        H = self.Hamiltonian(charge_offset = charge_offset, external_flux = external_flux)
        return eigsh(H, k = n, which = "SA")

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
    
    def n_matrix(self):
        return self.a_dagger() @ self.a()

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
        self.epsilon_I, self.epsilon_Q = self.envelopes()

        # Pulses
        self.I_pulse, self.Q_pulse = self.pulses()


    # Put together:
    def output(self):
        def I_pulse(t):
            I_comp = self.I * self.epsilon_I(t) * self.I_pulse(t)
            return I_comp
        
        def Q_pulse(t):
            Q_comp = self.Q * self.epsilon_Q(t) * self.Q_pulse(t)
            return Q_comp
        
        return I_pulse, Q_pulse


    def pulses(self):
        I_pulse = lambda t: np.cos(self.omega * t)
        Q_pulse = lambda t: np.sin(self.omega * t)
        return I_pulse, Q_pulse


    def envelopes(self):
        
        def epsilon_I(t):
            A = 1 / np.sqrt(2 * np.pi) / self.width
            return A * np.exp(-(t - self.T) ** 2 / self.width ** 2 / 2)
        
        if self.drag:
            def epsilon_Q(t):
                A = 1 / np.sqrt(2 * np.pi) / self.width
                B = - 1 / (2 * self.width ** 2) 
                return A * B * (2 * (t - self.T)) * np.exp(-(t - self.T) ** 2 / self.width ** 2 / 2)
        else:
            epsilon_Q = epsilon_I

        return epsilon_I, epsilon_Q


class ResonatorProbePulse():
    """
    This class creates a pulse that can be used to probe the resonator.
    It is made by having a fast rise and fall time using a sin^2 envelope. Inbetween it is constant.
    """

    def __init__(self, duration, omega, rise_time = 1, fall_time = 1, amplitude = 1, phase = 0):
        self.duration   = duration
        self.rise_time  = rise_time
        self.fall_time  = fall_time
        self.amplitude  = amplitude
        self.phase      = phase
        self.omega      = omega

        # Find I and Q
        self.I      = np.cos(phase)
        self.Q      = np.sin(phase)

        # Envelopes:
        self.epsilon_I, self.epsilon_Q = self.envelopes()

        # Pulses
        self.I_pulse, self.Q_pulse = self.pulses()


    # Put together:
    def output(self):
        def I_pulse(t):
            I_comp = self.I * self.epsilon_I(t) * self.I_pulse(t)
            return I_comp
        
        def Q_pulse(t):
            Q_comp = self.Q * self.epsilon_Q(t) * self.Q_pulse(t)
            return Q_comp
        
        I_pulse, Q_pulse = np.vectorize(I_pulse), np.vectorize(Q_pulse)

        return I_pulse, Q_pulse

    # Get the pulses
    def pulses(self):
        I_pulse = lambda t: np.cos(self.omega * (t - self.duration[0]))
        Q_pulse = lambda t: np.sin(self.omega * (t - self.duration[0]))
        return I_pulse, Q_pulse

    # Compute the envelopes
    def envelopes(self):

        def epsilon_I(t):
            if t >= self.duration[0] and t < self.duration[0] + self.rise_time:
                return self.amplitude * np.sin((t - self.duration[0])/self.rise_time * np.pi / 2 ) ** 2 
            elif t >= self.duration[0] + self.rise_time and t < self.duration[1] - self.fall_time:
                return self.amplitude
            elif t >= self.duration[1] - self.fall_time and t < self.duration[1]:
                return self.amplitude * np.sin((t - self.duration[1])/self.fall_time * 2 / np.pi ) ** 2
            else:
                return 0

        epsilon_Q = epsilon_I

        return epsilon_I, epsilon_Q


    # Rotating frame
    def rotating_frame(self, t, omega):
        """
        This function takes a time t and a frequency omega and returns the time in the rotating frame.
        It is given as a unitary transformation counteracting the driving by the probe pulse.
        """
        return np.exp(-1j * omega * (t - self.duration[0]))



if __name__ == "__main__":
    # res = Resonator(omega = 1)
    ts    = np.linspace(0, 100, 1000)
    probe = ResonatorProbePulse(duration = [20, 80], omega = 3, rise_time = 25, fall_time = 25, amplitude = 1, phase = 0)

    I, Q = np.vectorize(probe.epsilon_I), np.vectorize(probe.epsilon_Q)

    total_pulse = I(ts) + Q(ts)

    plt.plot(ts, total_pulse)
    plt.show()

