import numpy as np

# Mathematical tools:

# Define a function to take the partial trace of the density matrix

class System():

    def __init__(self, dimensions):
        self.dimensions = dimensions


    def partial_trace(self, density_matrix, trace_over = 0):
        # Add to check the dimensions. 
        decomposite = np.array(density_matrix).reshape(self.dimensions[0], self.dimensions[1], self.dimensions[0], self.dimensions[1])
        partial_trace = np.trace(decomposite, axis1 = trace_over, axis2 = trace_over + 2)
        return partial_trace
    
    def get_identities(self):
        I_1 = np.identity(self.dimensions[0])
        I_2 = np.identity(self.dimensions[1])
        return I_1, I_2
