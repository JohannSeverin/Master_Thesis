In this module, a **system** is made to simulate the interaction between different [[devices]] such that different interactions can be calculated. The system class also takes care of propagating updates directly to the devices which it is built of while also maintaining its own sweepable parameters.


## System Parent Class
The **System parent class** defines much of the logistics for the updating parameters in the overall system or in the device which it is made of. 

The parent class takes the following abstract form:

```python
class System(ABC):
    @abstractmethod
    def set_operators(self):
		WRITE THIS FUNCTION SUCH THAT IT UPDATES THE OPERATORS
		DEVICE OPERATORS ARE UPDATED BEFORE THIS FUNCTION IS CALLED

    @abstractmethod
    def get_states(self):
        THIS FUNCTION SHOULD BE USED TO GET THE BASIS STATES OF THE SYSTEM
```

In addition to the new methods, the new <code>__init__()</code> function should also define a <code>self.sweepable_parameters</code> and a <code>self.update_methods</code>. _Maybe also even a self.dimensions?_

An example for defining a new system can be seen here:


```python
class NewSystem(System):

	def __init__(self, qubit, PARAM1, PARAM2):
		CALCULATIONS HERE

		self.sweepable_parameters = ["PARAM1"]
		self.update_method = [self.update_operators, self.update_dissipators]

	def set_operators(self):
		CALCULATE HAMILTNONIAN FOR THE ENTIRE SYSTEM HERE

	def set_dissipators(self):
		CALCULATE THE DISSIPATORS HERE

	def get_states(self, state_numbers):
		states = FIND THE STATES HERE
		return states
	```


## Systems
Some simple systems are already defined in the module and are documented below. Some systems are approximation of these systems and will be found in the new section.

### QubitSystem
The simplest system connects a qubit to a pulse drive line. It can be defined by:

```python
QubitSystem(
	self,
	qubit: Device,
	qubit_pulse: Pulse = None,
)
```

To get a state, one can call the following code with state being the integer of the desired level. 

```python
state = QubitSystem.get_states(state: int)
```

A few simple methods are defined to get common expectation value operators.

```python
# An operator  for finding the number operator
QubitSystem.qubit_state_operator()

# Or the occupation for a specific state
QubitSystem.qubit_state_occupation_operator(state: int = 1)
```


### QubitResonatorSystem
The QubitResonatorSystem is made for combing one [[Device#Qubits]] class element with a [[Device#Resonator]] along with  [[pulses]] each. 

The QubitResonatorSystem is called with the following syntax:

```python
QubitResonatorSystem(
	qubit: Device,
	resonator: Device,
	coupling_strength: float,
	resonator_pulse: Pulse = None,
	qubit_pulse: Pulse = None,
)
```

The qubit and resonator are connected with the $g \; \hat{n} \otimes (a + a^\dagger)$ where $g$ is the coupling strength, $\hat{n}$ is the charge matrix of the qubit and $a$ and $a^\dagger$ are the lowering and raising operators of the resonator.

Initial states are found as $|\text{qubit state}\rangle$ $\otimes$ $|\text{resonator state}\rangle$ calling:

```python
QubitResonatorSystem.get_states(qubit_states: int = 0, resonator_states: int = 0)
```

And the following operators can be found to calculate common expectation values:

```python
# The photon number operator by tracing out the qubit
QubitResonatorSystem.photon_number_operator()

# The qubit number operator is found:
QubitResonatorSystem.qubit_state_operator()

# The occupation operator for a specifcic qubit state can be found  
QubitResonatorSystem.qubit_state_occupation_operator(state: int = 1)

# And the I and Q operator for measuring the quadratures of resonator can be found as
QubitResonatorSystem.resonator_I()
QubitResonatorSystem.resonator_Q()
```


## Approximated Systems
As some system very complex to simulate. For this reason a few approximations are made and implemented in order to get simpler simulations.

### DispersiveQubitResonatorSystem 
By taking the dispersive approximation of the [[#QubitResonatorSystem]] subject to a [[Pulses#Square Cosine Pulse]], one can do the dispersive approximation. The dispersive approximation, is most easily calculated by using the `.dispersive_approximation()` when a *QubitResonatorSystem* is defined with a *Square Cosine Pulse*.

As an example, the system can be defined by:

```python
QubitResonatorSystem(
	qubit: Device,
	resonator: Device,
	coupling_strength: float,
	resonator_pulse: Pulse = None,
	qubit_pulse: Pulse = None,
).dispersive_approximation(dispersive_shift: float = None)

```

where the `resonator_pulse ` must be a **SquareCosinePulse** and the **qubit_pulse** is ignored if defined. The DispersiveQubitResonatorSystem inherits the dissipators and stochastic dissipators from the QubitResonatorSystem, but redefines. One can give the function explicit dispersive shifts, otherwise it will be calculated using the frequencies of the qubit and the resonator together with the coupling strength. 

