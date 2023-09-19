---
code: /mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/code/calculations/decay_processes.ipynb;
  /mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/projects/in_measurement_calibration/evolution_of_long_pulse.ipynb
type: theoretical_calculations, experimental analysis
---
## Theory
In this note, the goal is to describe what the probability of being in either $|0\rangle$ or $|1\rangle$ if they decay to each other. Thus this system will not be driven and only subject to the Lindblad operators.

The Dynamics are 
$$
\dot{p_{0}}(t) = -\Gamma_{\uparrow}p_{0(t)} + \Gamma_{\downarrow}p_{1(t)}
$$
$$
\dot{p_{1}}(t) = \Gamma_{\uparrow} p_{0(t)} - \Gamma_{\downarrow}p_1(t)
$$
Where $\Gamma_\uparrow$ is the rate where $0\to 1$ and $\Gamma_\downarrow$ the rate where $1 \to 0$. Thus the change is just what comes in from the other state minus the amount that leaves.

Initially the system is in some superposition of the two states depending on the initialization. 
$$
\rho_{init}= p_{0}(0)\ |0\rangle\langle0| + p_{1}(0) \ |1\rangle \langle1|
$$
Where $p_{0} + p_{1} = 1$ (normalized in occupation probability). 

If we were to initialize into $p_{1}= 1$ but with some small infidelity $f_0$, we would have $p_{0}=f_{0;}\quad p_{1} = 1- f_0$ . using this as initial for solving the coupled differential equation, we find the solution is:

$$
p_{0}(t) = \frac{\Gamma_\downarrow}{\Gamma_\uparrow + \Gamma_\downarrow}-\left( \frac{\Gamma_\downarrow}{\Gamma_\uparrow + \Gamma_\downarrow} - f_{0}\right)e^{-t(\Gamma_\uparrow + \Gamma_\downarrow)}
$$
and 
$$
p_{1}(t) = \frac{\Gamma_\uparrow}{\Gamma_\uparrow + \Gamma_\downarrow}+\left( \frac{\Gamma_\downarrow}{\Gamma_\uparrow + \Gamma_\downarrow} - f_{0}\right)e^{-t(\Gamma_\uparrow + \Gamma_\downarrow)}
$$

Which can in the limits are:
- For $t\to 0;\quad e^{-t(\Gamma_\downarrow+\Gamma_{\uparrow})}\to 1$ such that $p_{0} \to f_{0}$ and $p_{1}\to 1-f_0$ as originally intended in the initial conditions.
- For $t\to \infty;\quad e^{-t(\Gamma_\downarrow+\Gamma_{\uparrow})}\to 0$ and the steady state is achieved with steady:
	- $p_{0} \to \frac{\Gamma_\downarrow}{\Gamma_{\uparrow}+ \Gamma_\downarrow}$ 
	- $p_{1} \to \frac{\Gamma_\uparrow}{\Gamma_{\uparrow}+ \Gamma_\downarrow}$ 

![[time_evolution_tls.png]]

## Experiment
We use the long measurements found from the experiment in [[First Control of ADC Trace Data]].

First we take the first $\approx 1.0 \;\mu s$ we train a Linear Discriminator. The decision boundary is the one seen below.  It separates pretty nicely the two groups.

![[training_LDA.png]]

If we look at the animation which is demodulated in $1000 \text{ ns}$ intervals we can show the evolution as an animation of the IQ blobs. We clearly see the excited states heading to the ground state blob.

![[animation_of_data.mp4]]

If we use the trained LDA to classify points at every step. We find the following:

[[evolution_of_two_states.png|Open: Pasted image 20230722144715.png]]
![[evolution_of_two_states.png]]

Which looks pretty close to the theory presented in the top of the note.

Next step is fitting the data with the function from the beginning. We reformulate them as:

```python
def ground_func(x, steady_state, initial_0, decay_time):
    return steady_state - (steady_state - initial_error) * np.exp(-x / decay_time)

def excited_func(x, steady_state, initial_0, decay_time):
    return steady_state + (1 - steady_state - initial_error) * np.exp(-x / decay_time)
```

![[Fit_of_evolution.png]]

Where we get the following parameters from the fit:

|Parameter|Ground Fit|Excited Fit|
|----|----|----|
|Steady state|$0.951 \pm 0.254$|$0.099 \pm 0.004$|
|initial_0|$0.875 \pm 0.003$|$0.124 \pm 0.008$|
|decay_time|$(159 \pm 615) \text{ µs}$|$(11.1 \pm 0.3)  \text{ µs}$|

Decay from calibration is $9.9 ± 0.5 \text{ μs}$



