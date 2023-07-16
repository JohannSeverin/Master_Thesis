#todo 
- [ ] Write the entire table of contents

All the physical devices and pulses are written as children to the <code>Device</code>-class. 

All devices are collected in three main categories:

- **[[Device]]** store the physical devices with Hamiltonian, decays and other parameters
- **[[Systems]]** are connections of physical devices. 
- **[[Pulses]]** are different time dependent pulses which can be coupled at the appropriate keyword in a [System](Systems)


## List of Devices:
A running list of devices, systems and pulses are found here:

- [[Device]]
	- [Simple Qubit](Device#SimpleQubit)
	- [Transmon](Device#Transmon)
	- [Resonator](Device#Resonator)
- [[Systems]]
	- [Qubit System]
	- [Qubit Resonator System]
	- [Approximated Systems]
- [[Pulses]]
	- [GaussianPulse](Pulses#GaussianPulse)
	- [SquareCosinePulse](Pulses#SquareCosinePulse)
	- [CloakedPulse](Pulses#CloakedPulse)


## Device Parent Class

The Device Parent Class:

```python
Device(ABC):

	def set_operators(self) -> None:
		SHOULD BE OVERWRITTEN TO SET DEVICES OPERATORS GIVEN PARAMETERS

```

is an abstract class made to keep track of static and sweepable parameters. It has the following methods. When subclassed it should have a new version which creates a new init calling the parent and overwriting the <code>Device.set_operators</code> method. 

The <code>__init__(self)</code> should define a <code>self.sweepable_parameters</code> a list with strings referring to the defined parameters which should have the ability to be swept in an [[Experiment]]. Furthermore, it should also include <code>self.update_methods<//code> 

An example could be the following:


```python

def NewDevice(Device):

	def __init__(self, ...):
		DEFINE NEW PARAMETERS HERE FOR 
		self.sweepable_parameters = ["PARAM1", "PARAM2"]
		self.update_methods = [self.set_operators]
		super().__init__()

	def set_operators(self):
		SET HAMILTONIAN AND OTHER OPERATORS HERE
		self.hamiltonian = qutip.number(self.levels) # For Example a harmonic oscillator
	
```

