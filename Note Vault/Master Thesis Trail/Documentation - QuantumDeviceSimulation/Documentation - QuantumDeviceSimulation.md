---
sticker: ""
tag: frontmatter
---
This is the documentation for a simulation tool created as part of my Master Thesis.  Ultimately, this module is built as a wrapper for [QuTiP](https://qutip.org/) in an attempt to ease simulation of routine tasks in Superconducting Qubits. 

The module is built up of three main parts: [[Devices]], [[Simulation]] and [[Analysis]] which are working together to built, simulate and analyze the desired superconducting system.

The complete content of this module:

---
## [[Devices]]

To built up the devices and quantum chips, we have all statics in the device part of the module.  The main goals are to calculate and store:
* Hamiltonians from calibrated or device parameters
* Decoherence operators
* Interaction between different devices in a so called [[Devices/System]]
* Different pulses which can be send in to the devices to interact with them

---
## [[Simulation]]

When devices are designed, the time evolution can be calculated at different degrees of complexity. These simulation strategies are found in this part of the module. And currently support the following:
- Unitary Evolutions using the Schrodinger Equation
- Lindblad Evolution which also take decoherence into account by evolving the total density matrix and collapse operators
	- This can be done by using the Lindblad Master Equation to do deterministically
	- Or by doing a Monte Carlo Style Experiment

--- 
## [[Analysis]]

Lastly, we have a module to do common analysis of the simulation traces. This is still very much under development. The hope is, however, that it should take xarrays of the same type support in [OPX Control](https://github.com/cqed-at-qdev/OPXControl/tree/main/opx_control). Hopefully this will decrease the distance between simulations and experiment to hopefully integrate the two together. 



