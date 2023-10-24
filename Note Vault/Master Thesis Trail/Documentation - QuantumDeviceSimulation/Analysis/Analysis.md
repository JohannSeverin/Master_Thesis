---
tags: []
---
The analysis module serves as a convenient way for plotting results from the Simulations class. It is in a preliminary stage and is mostly used for overview and debugging purposes. At the current stage, it is built of two groups.

* [[#Sweep Analysis]] - Which serves as way of plotting different sweeps or time dependent behavior from a simulation. The different sweep-analysis methods can be chosen automatically by using `automatic_analysis` from `analysis.auto`. 
* [[#Q Function Analysis]] - Is used to calculate and visualize the Q function for density matrices from simulations.


## Sweep Analysis 
At the moment, the sweep analysis have four functions which both support multiple initial states and multiple expectation values, which will be shown in a grid.
* `plot_one_dimensional_sweep(results: SimulationResults, **kwargs)`, which takes  a `SimulationResults` object with one sweep parameter and plots the expectation values against it.
* `plot_two_dimensional_sweep(results: SimulationResults, **kwargs)`, which takes  a `SimulationResults` object with two sweep parameters and plot a heatmap with the expectation values as function of both. 
* `plot_time_evolution(results: SimulationResults, **kwargs)`, which takes  a `SimulationResults` object with no sweep parameters, but with `only_store_final = False` and plots the time-dependence of the expectation values. 
* `plot_time_evolution_with_single_sweep(results: SimulationResults, **kwargs)`,  which takes  a `SimulationResults` object with one sweep parameter and `only_store_final = False`. It then plots the a heatmap of the expectation values where the axis are the sweep parameter and time.

Instead of choosing, one can use the automatic analysis:

```python
# With results from some experiment 
results = experiment.run()

from analysis.auto import automatic_analysis
automatic_analysis(results)
```

which automatically detects the amount of sweep parameters and if a time-axis is available to determine which of the sweep plots above should be shown. 

## Q Function Analysis
The Q Function Analysis consists of one utility function:

```python
Q_of_rho(
	 rhos: iterable[qutip.Qobj], 
	 x: np.ndarray, 
	 y: np.ndarray, 
	 rotate: iterable[float] = 0
 )
```

Which takes a list of state along with a list of x, y coordinates where the q function should be calculated for these states. To support demodulation behavior a rotation amount can be given in radians for each of the states. This will rotate the x-y coordinate system with the desired amount.

And two plotting functions:
* `qfunc_plotter(results: SimulationResults, interval=10, resolution=100)` which takes a simulation result and plots the Q function of the resonator after tracing out the qubit state.
* `qfunc_plotter_with_time_slider(results: SimulationResults, interval=10, resolution=100, time_steps=1, demod_frequency=0)` which plots the time-dependent q function with slider which chooses at what time the Q function should be displayed.