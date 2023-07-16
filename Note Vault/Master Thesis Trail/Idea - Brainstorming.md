_Just brainstorming the final idea for the project_

The Idea is to represent the problem as the differential equations and then integrate it using PyTorch or Tensorflow.

## Software
- __Tensorflow:__ I'm used to, so that would be easy to set up. It does not support Stochastic Differential Equation yet though. [OdeSolver](https://www.tensorflow.org/probability/api_docs/python/tfp/math/ode/Solver)
	- It supports complex numbers
- __PyTorch:__ This will require a bit more learning on my site. It however has a stochastic differential equation solver 
	- For Ordinary Differentail Equation, we can use [torchDiffEq](https://github.com/rtqichen/torchdiffeq)
	- For Stochastic we use [torchSDE](https://github.com/google-research/torchsde)


## ODE Version
For the ODE version, we will have to use Lindblad Master Equation and write it in a super operator formalism. 
- Use regular ODE solver.
	- Take the Lindblad Master equation to calculate the evolution of the initial state
	- Calculate the Q-function at each time 
	- Use $-\log(Q(\rho(t)))$ as the loss function for points.
	- We can now minimize the parameters for the fit.
- The quantum efficiency will have to be propagated to the loss function.
	- Maybe fit parameter, otherwise we take this as a calibration parameter.
- Possible Fit Parameters
	- Hamiltonian / Drive.
	- Dispersive shift
	- __Decoherence__
		- T1 decay
	- State Prep Error (start with a weak initial X gate). Fit the fidelity of this one
	- _Measurement Efficiency_


## SDE Solver 
#todo
- [ ] Fill out SDE solving ideas




