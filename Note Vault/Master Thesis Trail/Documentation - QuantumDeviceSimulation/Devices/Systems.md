#todo 
- [ ] Write documentation for the defined systems
	- [ ] Qubit System
	- [ ] Qubit Resonator System
	- [ ] The approximated Systems


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


### QubitResonatorSystem



## Approximated Systems

### DispersiveQubitResonatorSystem