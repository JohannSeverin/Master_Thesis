---
code: /mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/projects/photon_calibration/data_exploration.ipynb
date: 2023-07-19
---
Some preliminary look at the data from [[Data from Experiments]]. 

## Dispersive Shift
We take data from the double resonator and fit it with a gaussian dip. 
![[dispersive_dips.png]]

From this we also pick up:
- The coupling from $\chi = g^2 / \Delta \to g = 33.6$ MHz
	- This can also be checked by the frequency of vacuum rabi oscillations
-  Critical Photon Number can be extracted from: $n_{crit} = \left(\frac{\Delta}{2g}\right)^2 = 547$

## Pulse First 
Plotting the two data with scaling of X-pulse amplitude of $0$ and $1$ we obtain the following. (This is data from [[Data from Experiments#Data]])

![[raw_data.png]]

To find the actual Qubit Frequency at a given amplitude we take the difference between the two:

![[difference.png]]

We now look at each column on the data and the associated standard error found by averaging the data. We fit a const minus a gaussian to find the dip at each amplitude. Along the $0$ drive amplitude it looks like:
![[example_dip.png]]

Doing this for all the columns we find the dip and uncertainity at every drive ampltiude. We summarize it in the following figure:

![[dip_as_function_of_frequency.png]]

## Converting Drive Frequency dip to Photon Number
Considering the dispersive Hamiltonian of the qubit-resonator-system, we have above primarily considered the shift of resonator frequency depending on the qubit state, but just moving the parenthesis to represent it in the form:

$$H = \tilde{\omega}_r a^\dagger a + \left(\frac12 \tilde{\omega}_{01} + \chi a^\dagger a\right) \sigma_z$$
making it apparent that while the qubit state moves the resonator, the opposite is also true. Since the qubit frequency moves with $\chi$ for each photon in the resonator, the photon number at a given amplitude can be calibrated

From [[#Dispersive Shift]] we found $2\chi = 1.44 \pm 0.04$ MHz $\Rightarrow$ = $\chi = 0.72 \pm 0.02$ MHz. we consider the photon number to be low  $\to$ there is no second higher order terms, so we consider the relation between "qubit drift" and photon number to be linear. 
![[Calculated_drift.png]]

We can now divide the qubit drift with the dispersive shift to find:
![[Photon_Count.png]]


## In Measurement X-Gate
Will probably be better when taking the following data:
- [ ] Check the dependence on time.
	- [ ] Are we in steady state
	- [ ] Vacuum Rabi Oscillations for second opinion on the coupling
- [ ] Probably longer pulses in the measurement data
- [ ]Drive power .> $\sqrt{A}$
- 