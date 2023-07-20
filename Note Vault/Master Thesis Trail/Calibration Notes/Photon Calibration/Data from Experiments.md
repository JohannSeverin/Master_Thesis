---
date: 2023-07-19
---
Considering the dispersive Hamiltonian of the qubit-resonator-system, we have above primarily considered the shift of resonator frequency depending on the qubit state, but just moving the parenthesis to represent it in the form:

$$H = \tilde{\omega}_r a^\dagger a + \left(\frac12 \tilde{\omega}_{01} + \chi a^\dagger a\right) \sigma_z$$
making it apparent that while the qubit state moves the resonator, the opposite is also true. Since the qubit frequency moves with $\chi$ for each photon in the resonator, the photon number at a given amplitude can be calibrated.

## Pre Calibration
Before starting, we calibrated:
- Resonator spectroscopy for $|0\rangle$ and $|1\rangle$
	- Used for finding the dispersive shift
- A proper $X$-pulse was also calibrated using drag/Rabi/Qubit Spectroscopy
- T1

### Resonator Spectroscopy 

![[Readout_frequency_152644.png]]
​
### Rabi
![[Rabi_160133.png]]
### $T_1$

![[T1_152810.png]]





## Data 
Two datasets were taken for this:
1.  One were a pulse was sent unto the Resonator with a given amplitude. An X-pulse was sent in the end of the pulse and measurement were performed.
![[photon_calibration_155708.png]]

2. The other one the X-pulse was send doing the readout pulse. Half way through the 1 µs pulse. 
![[photon_calibration_in_readout_164131.png]]


> [!NOTE]
> Question: Why are these two curves of different length? Do we not reach steady state after the $500 ns$?
> The first one is filled for $1000 ns$ before the X-pulse and the other for $500 ns$. I though it would have been in steady state by then. Maybe should just do a scan over duration in the amplitude in question. 

