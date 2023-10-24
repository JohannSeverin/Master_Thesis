The simulation is the module responsible for the time integration. Most of the functionality is collected in the `SimulationExperiment` - parent class which keeps track of sweeping, data storage, calculation of expectation values etc.. The `SimulationExperiment` is subclassed in order to provide a method for simulating. This allows us to to simulate using different models like Unitary, Lindblad or Stochastic simulations. 

The following subclasses are available:
- [[#Schrödinger Experiment]] allows unitary evolution without loss. This is however one dimensional and can go very fast.
- [[#Lindblad Experiment]] is a deterministic evolution using the Lindblad Master Equation to take care of decoherence and leakage due to interaction with the environment
- [[#Monte Carlo Experiment]] takes care of decoherence and losses to the environment by applying them stochastically. This means that we can approximate the Lindblad solution by dialing up the number of trajectories.
- [[#Stochastic Master Equation]] is used for simulating homodyne and heterodyne measurements of the system and how it behaves under continuous monitoring.

# Simulation Experiment Class
> [!NOTE]
> This class uses a  `Dataclass` to store data from simulation and save it. In the future this should be changed such that it comes in the same way as from the ´OPX_control´ library which is used in the lab.  

The `SimulationExperiment` call the overwritten `simulate` method to simulate a configuration. This now takes care of looping over the swept parameters defined in the [[Systems]] simulated. At the moment it supports sweeps over 1 or 2 parameters as well as the possibility to save the state or density matrix at each time in the simulation.

# Subclasses
## Schrödinger Experiment
The simplest implemented experiment is the Schrödinger experiment which takes states and simply evolves them using the Schrödinger equation:
$$\frac{d}{dt}\ket{\psi} = -i\hbar\hat{H}\ket{\psi}$$
The Schrödinger equation is unitary evolutions and does not support decoherences, density matrices or measurements. 

Since the Schrödinger equation is unitary, it is not necessary to keep track of a whole density matrix and is for this reason the fastest of the possible simulations.

To run an experiment it can be defined using:
```python
experiment = SchoedingerExperiment(
	system: System,
	states: Iterable[qutip.Qobj],
	times: Iterable[float],
	expectation_operators: list[qutip.Qobj] = [],
	store_states: bool = False,
	store_measurements: bool = False,
	only_store_final: bool = False,
	save_path: str = None,
)
```

And can then be run simply by:
```python
results = experient.run()
```

The parameters of the `SchroedingerExperiment` are as follows:

|Parameter|Function|
|----|----|
|system|The System that should be simulated|
|states|States or list of states to simulate|
|times|List of times for simulation|
|expectation_operators|List of operators for which an expectation value should be calculated|
|store_states|Whether the states should be stored|
|only_store_final|Whether only the final state should be considered in storing states and calculating expectation vaues|
|save_path|What path the final results should be saved to|

## Lindblad Experiment
To consider losses and decoherences from the system, one should use the Lindblad Master equation to simulate the time evolution of a density matrix in contact with the environment. The master equation which is evolved is given by:
$$    \dot{\rho}(t) = -i dt[H, \rho(t)] +  \sum_a \left(L_a \rho(t) L_a^\dagger - \frac12 L_a L_a^\dagger \rho(t) - \frac12 \rho(t)L_a L_a^\dagger  \right)$$
where $L_\alpha$ is dissipation operators.

The Lindblad Master Equation considers a linear equation for the density matrix and for this reason it scales heavy with the size of the Hilbert Space. For this reason, it is significantly slower than the Schrödinger equation for high dimensional problems.

To simulate the Lindblad Master equation the following experiment should be set up in exactly the same way as the `SchoedingerExperiment`, but will consider dissipation operators of the system:

```python
experiment = LindbladExperiment(
	system: System,
	states: Iterable[qutip.Qobj],
	times: Iterable[float],
	expectation_operators: list[qutip.Qobj] = [],
	store_states: bool = False,
	store_measurements: bool = False,
	only_store_final: bool = False,
	save_path: str = None,
)
```

And can be run by:

```python
experiment.run()
```

The parameters are also the same as `SchroedinerExperiment`:

|Parameter|Function|
|----|----|
|system|The System that should be simulated|
|states|States or list of states to simulate|
|times|List of times for simulation|
|expectation_operators|List of operators for which an expectation value should be calculated|
|store_states|Whether the states should be stored|
|only_store_final|Whether only the final state should be considered in storing states and calculating expectation vaues|
|save_path|What path the final results should be saved to|

## Monte Carlo Experiment
> [!NOTE]
> While the Monte Carlo method uses parallel processes, it is done at a python level and runs inefficient. This method should instead be changed like the [[#Stochastic Master Equation]], where qutip handles the parallel calls internally. 

For larger dimensions it can be beneficial to add the collapse operators stochastically by using the Monte Carlo Simulation. This simulation will apply a collapse operator depending on the given rate and time step. By doing this, the problem is still one dimensional and can be repeated multiple times to approximate the Lindblad equation. 

A Monte Carlo Experiment is setup using:

```python
experiment = MonteCarloExperiment(
	system: System,
	states: Iterable[qutip.Qobj],
	times: Iterable[float],
	expectation_operators: list[qutip.Qobj] = [],
	store_states: bool = False,
	store_measurements: bool = False,
	only_store_final: bool = False,
	save_path: str = None,
	ntraj: int = 1,
	exp_val_method="average",
)
```

And is run by:

```python
results = experiment.run()
```

The parameters of the `MonteCarloExperiment` are:

| Parameter             | Function                                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| system                | The System that should be simulated                                                                                                 |
| states                | States or list of states to simulate                                                                                                |
| times                 | List of times for simulation                                                                                                        |
| expectation_operators | List of operators for which an expectation value should be calculated                                                               |
| store_states          | Whether the states should be stored                                                                                                 |
| only_store_final      | Whether only the final state should be considered in storing states and calculating expectation vaues                               |
| save_path             | What path the final results should be saved to                                                                                      |
| ntraj                 | How many times to repeat each subexperiment                                                                                         |
| exp_val_method        | When set to "average" this will take the average of all trajectories other wise expectation values for each trajectory is returened |
## Stochastic Master Equation
The most complex experiment is evolving the stochastic master equation. In addition, to including decoherence and dissipation this also allows for measurement feedback and a measurement record. 

The Stochastic Master Equation takes the form:
$$    d\rho = -i [H, \rho]dt + \mathcal{D}[c]\rho dt + \mathcal{H}[c] \rho dW$$
Where the superoperators refer to the Lindblad dissipator:
$$\mathcal{D}[c]\rho(t) = c \rho(t) c^\dagger - \frac12 c c^\dagger \rho(t) - \frac12 \rho(t) cc^\dagger$$
And a stochastic part given by:
$$\mathcal{H}[c]\rho(t) = c\rho(t) + \rho(t)c - \langle c + c^{\dagger}\rangle \rho(t)$$
Here $dW$ is a stochastic variable of the wiener process with variance $dt$.

Currently this is simulated either by using a homodyne or heterodyne setup of the collapse operator using the `method` keyword, depending on whether one or two quadratures should be measured. 

When using `StochasticMasterEquation` the system parameter `system.stochastic_dissipators` will also be considered and add the stochastic term with weighted by the efficiency of the system: `system.readout_efficiency`.

The results of the `StochasticMasterEquation` will include `measurements`. The measurements are the result of a record of outcomes from the measurement at each timestep. In the heterodyne measurements (which is the only one implemented), the measurement record takes the form of:
$$    dr = \eta \left(  \langle I \rangle + i\langle Q \rangle\right) dt + \frac{dW_{I} + dW_{Q}}{\sqrt{2}}$$
The experiment is defined using:
```python
experiment = StochasticMasterEquation(
	system: System,
	states: Iterable[qutip.Qobj],
	times: Iterable[float],
	expectation_operators: list[qutip.Qobj] = [],
	store_states: bool = False,
	store_measurements: bool = False,
	only_store_final: bool = False,
	save_path: str = None,
	ntraj: int = 1,
	exp_val_method="average",
	method: str =  "heterodyne",
	store_measurements: bool = True,
	nsubsteps: int = 1,
)
```

And run like the others by:

```python
results = experiment.run()
```

The parameters are:

| Parameter             | Function                                                                                                                            |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| system                | The System that should be simulated                                                                                                 |
| states                | States or list of states to simulate                                                                                                |
| times                 | List of times for simulation                                                                                                        |
| expectation_operators | List of operators for which an expectation value should be calculated                                                               |
| store_states          | Whether the states should be stored                                                                                                 |
| only_store_final      | Whether only the final state should be considered in storing states and calculating expectation vaues                               |
| save_path             | What path the final results should be saved to                                                                                      |
| ntraj                 | How many times to repeat each subexperiment                                                                                         |
| exp_val_method        | When set to "average" this will take the average of all trajectories other wise expectation values for each trajectory is returened |
| method                | Set to "homodyne" or "heterodyne" depending on one or two quadrature measurement                                                    |
| store_measurements    | whether to store the measurement record                                                                                             |
| nsubsteps             | If there should be substeps in the simulation between returned points                                                               |