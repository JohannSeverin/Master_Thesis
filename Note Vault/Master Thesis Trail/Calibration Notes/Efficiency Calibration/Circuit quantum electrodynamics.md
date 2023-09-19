This is a larger article which is good for review. At the moment, we will consider the results for qubit dephasing by resonator filling in order to setup a calibration scheme for determining the quantum readout efficiency.

## Main Results:

The main result is that the quantum efficiency $\eta$ of the entire readout chain can be found by using the Signal to Noise Ratio $\text{SNR}$ together with a measure of dephasing during the readout time given by:
$$\beta_{m}= 2\chi \int_0^{\tau_{m}}dt \Im[\alpha_{g}(t) \alpha_e^*(t)]$$
These are combined to give:
$$\eta = \frac{\text{SNR}^2}{2\beta_m}$$


## Measurement induced dephasing
The effects here are very similar to the ideas used in [[Photon Calibration]] were the shift of the qubit from the resonator photon are used together with knowledge about $2\chi$ to retrieve the photon count.

[[Circuit quantum electrodynamics fig25.png|Open: Pasted image 20230817114351.png]]
![[Circuit quantum electrodynamics fig25.png]]

We can however also use another effect of the interaction, namely that the resonator also adds a broadening to the qubit spectroscopy width. Since the interaction is given by $2\chi a^\dagger a$ the broadening happens since the higher coherent states are build up of a combination of multiple higher occupied Fock states.  

