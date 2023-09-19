**Primarily this is taken from A Quantum Engineers guide. It is however modified to fit with the section where things are formulated in terms of Lindblad Operators.**

If we consider a qubit in the picture of the Bloch Sphere, we have the $\ket{0}$ at the north pole and $\ket{1}$ at the south pole. In this picture, we consider two types of decoherence. One along longitudinal direction (the $z$-axis), and the transverse (in the $x, y$ - plane). 

## Longitudinal Relaxation 
The longitudinal decay comes from transitions from $\ket{1} \leftrightarrow \ket{0}$  by coupling to the qubit in either the $x$ or in the $y$ channel. This could be from energy exchange to and from the environment. 

*If we were to sum over the coupling operators of the from creating or annihilating a particle (or quasi particle) in the environment at the cost of doing the opposite on the qubit. 
$$\sum\limits_{i}\Gamma_{i_\uparrow}\left(\sigma_{+}a_{i}\right) + \sum\limits_i\Gamma_{i\downarrow}( \sigma_{-}a_i^{\dagger})$$
If we were to trace out the environment now, we would be left with Lindblad operators of the type $\mathcal{L}[\sigma_{+}]$ or $\mathcal{L}[\sigma_{-}]$ which are connecting to the $x-y-$channels. Thus driving transitions relaxing the qubit from $\ket{1}\to\ket{0}$ or absorbing energy from the environment to go from $\ket{0}\to\ket{1}$. 

However, since the state $\ket{1}$ has a higher energy $\Delta E=\hbar\omega$ the excitation rate $\Gamma_{\uparrow}$ is lower compared to the relaxation rate $\Gamma_\downarrow$ for low temperatures $\tau\lessapprox\omega_{01}$, The Boltzmann factor relating the two rates are given by: $\exp(\omega/\tau)$ so for low temperatures $\approx0-50 \text{ mK}$ the steady state of the qubit will be the ground state and the total longitudinal decay rate will be $\Gamma_{1} = \Gamma_\downarrow + \Gamma_{uparrow}\approx\Gamma_\downarrow$. Often this rate is converted to a time as a measure of the stability of the qubit. Here it is defined by $T_1=1/\Gamma_1$. 


## Dephasing

> [!NOTE]
> This can probably be formulated better with the use of "Simple Derivation of the Lindblad Equation" where they introduce the random unitary operator to start phases change
> Probably redo this section when writing the calibration notes!
> T2 is probably not important in our scheme since the connection between qubit and resonator is in charge basis on not longitudinal.

If qubits instead connect longitudinally (along the $z-axis$) they can alter the qubit frequency $\omega_{01}$ which setts us at a disadvantage. Since we normally think about the $x-$ and $y-$axis in the rotating frame with the frequency $\omega_{01}$ any changes to the qubit frequency would speed up / slow down the actual rotation, and ultimately we lose information about the actual phase of the qubit. 

Since there is no energy exchange with the environment this is a unitary process, and it is theory possible to reverse the effect and place the qubit back in the reference frame we know. However, this would assume that we have complete information about the time-dependence of the effective qubit frequency which is not realistic. With a clever use of gates, we can however decouple the pulse by using dynamical decoupling schemes which we will shortly return to in the section about calibrating the characteristic dephasing time $T_2$. *The concept is simply to apply $X_{\pi}$ pulses frequently to refocus the noise. Thus if some qubit initializations precess faster or slower adding a $X_\pi$ gate would flip the order allowing the fast ones to cast up and the slow ones to fall back to the actual qubit frequency

The rate of dephasing has two components, one from the stochastic "pure" dephasing time described above. This rate is given as $\Gamma_\phi$. The second contribution comes from energy relaxation since any superposition would lose all phase information once collapsed. Think of the $\frac{1}{\sqrt(2)}\left(\ket{0} + e^{i\phi}\ket{1}\right)$ superposition. If $\ket{1}$ were to decay to $\ket{0}$ the phase of the qubit state would also be lost. The total dephasing rate can be found by:
$$\Gamma_{2}=\Gamma_\phi+\frac{\Gamma_{1}}{2}$$
and the characteristic dephasing time is given by:
$$T_2=\frac{1}{\Gamma_{2}}$$

