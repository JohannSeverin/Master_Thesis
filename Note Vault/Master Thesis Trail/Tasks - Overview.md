---
tag: todo
sticker: lucide//check-square
---
Quick overview of things to do. The overview is made the [[2023-07-18]]. 

The goal post is set approximately, so all the [[#Large Tasks]] and [[#Travel Tasks]] are done by the [[2023-09-01]] and the [[#Projects]] should be started by then and completed by [[2023-10-01]], such that there will be an entire month of writing left.
 

# Tracking
## Latest Added
```dataview
TASK
WHERE !completed
SORT rows.created ASC
LIMIT 10
```

## Latest Finished
```dataview
task
where completed
limit 10
sort rows.ctime ASC
```
# Tasks

## Essential
- [ ] Get the soprano Qubit 1 characterized and bring it into simulation, so we can run the same experiments in both simulation and experiment.
- [ ] Do common calibration examples such that they are the same.
	- [ ] Spectroscopy
	- [ ] Rabi
	- [ ] Decays 
		- [ ] T1 
		- [ ] Ramsey
		- [ ] Echo
- [ ] Do the newer calibrations:
	- [ ] [[Efficiency Calibration]] 
	- [ ] [[Photon Calibration]] 
	- [ ] In-measurement T1
- [ ] Readout Strategies
	- [ ] Kick to $\ket{2}$ before reading out
	- [ ] Counter Pulse to depopulate the readout
## Ambitious
- [ ] Fitting the entire Master Equation / Stochastic Master Equation. See [[Idea - Brainstorming]]
- [ ] Write common operations on data to function in the analysis module
## Extras
- [ ] Cloaking
	- [ ] Relook the theory
	- [ ] Start implementing
		- [ ] in simulation
		- [ ] in experiment

## Writing
Look overleaf document.

- [ ] Writing Tasks 
	- [ ] **Qubit Decoherence. Can probably be split in smaller tasks, but use a day to get going.**
		- [ ] Include plots of decay from $|1\rangle \to |0\rangle$ 
		- [ ] Purcell
		- [ ] Resonator
	- [ ] Measurements. Find good sources and write about demodulation, homodyne and heterodyne measurement schemes.
	- [ ] Calibrations. Same, do a big push of writing these out with proper simulations from the [[Documentation - QuantumDeviceSimulation/Documentation - QuantumDeviceSimulation]]
	- [ ] Experimental Setup should be covered at least in some detail so it can be written later. Theory about Amplificication and Cooling might be important, so cover that in detail now.* 

