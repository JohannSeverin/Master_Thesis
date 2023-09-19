In this paper, a method for extracting the quantum efficiency, $\eta$ is proposed and demonstrated. 



## Outline
Three step plan:
1. Include a tune-up / tune-down of the resonator occupation of photon. Make sure to calibrate this to have optimal weights for the quadratures. 
2. Obtain the measurement-induced dephasing applied to the qubit by the photons. This is done by including the "readout-pulse" in between a Ramsey style experiment. 
3. Measure the Signal to Noise Ratio of the variable-strength weak measurement. 


## Theory
*I'll return and write this in a bit more detail, but have to dive a bit deeper in to the supplemental material*

## Method
In order to do the experiment, we need to compare the signal to noise ratio with the measurement induced dephasing of the qubit which serves as a measure of the measurement backaction on the qubit.

### Step 1: Deriving a Proper Pulse and Weights
We need to create a pulse which starts and ends at 0. This can be done by driving it and then waiting or alternatively do an active resonator reset. *As a starting point, we will probably just wait 3 or 4 times the ring-down period, but the resonator reset should definitely be explored also to do active reset.*

We then need to get the optimal linear weights, which in the gaussian picture will simply be performed by finding the linear weights.

In the paper, the weights look like:
![[General method for extracting the quantum efficiency of dispersive qubit readout in circuit QED.png]]

### Step 2: Finding the Dephasing by measurement
To get the dephasing rate, we can pack the pulse into a Ramsey-style experiment. Looking at the density matrix, we can start with:

$$X_\frac{\pi}{2}\ket{0}\bra{0}
= \begin{pmatrix} \frac{1}{2} & \frac{1}{2} \\ \frac{1}{2}& \frac{1}{2} \end{pmatrix}$$
Now doing the measurement we will diminish the coherence $\rho_{01}, \rho_{10}$ with a certain rate. We can determine the decoherence of the final signal by rotating again with $R_{\phi}^{\pi/2}$ to retrieve a plot looking something like:

![[General method for extracting the quantum efficiency of dispersive qubit readout in circuit QED-1.png]]

Where the coherences are extracted from the amplitude of the cosine pulse. And the figures from left to right refer to the amplitude of the pulse. 

### Step 3: Measuring the SNR
The Signal to Noise ratio is measured by using the resonator pulse from above as a readout pulse. Thus we initialize the pulse as either $\ket{0}$ or $\ket{1}$ by applying or omit an $X$-gate. The signal from the pulse is now measured and demodulated. The separation between and width of each of the peaks for $\ket{0}$ and $\ket{1}$ are calculated to get the SNR as:
$$
S=\left|\langle V_{\mathrm{int,}|1\rangle}-V_{\mathrm{int,}|0\rangle}\rangle\right|; \quad
N^{2}=\langle V_{\mathrm{int}}^{2}\rangle-\langle V_{\mathrm{int}}\rangle^{2}. 
$$
$$\to \text{SNR} = S/N$$
**Need to check if it's: $N^{2}= N_\ket{0}^2+N_\ket{1}^2$ or if it just a collected version (or if that is in total the same)**

The process is summarized in the following figure:
![[General method for extracting the quantum efficiency of dispersive qubit readout in circuit QED-2.png]]


## Extrapolating the results
When this method is repeated for multiple driving amplitudes of the pulse, we can generate get a graph like the following:
![[General method for extracting the quantum efficiency of dispersive qubit readout in circuit QED-3.png]]

Since the SNR is given as $\text{SNR} = a\epsilon$ and  $|\rho_{01}|=b\exp(-\frac{\epsilon^{2}}{2\sigma^{2}})$ doing the fits give good measures for $\sigma$ and $a$ which are need for calculating the quantum efficiency found to be:
$$
\eta=\frac{a^{2}\sigma^{2}}{2}
$$

