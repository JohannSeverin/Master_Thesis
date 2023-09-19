Device is part of the Devices submodule which contains the different children classes to the [Device Parent Class](Devices#Device Parent Class).  

## Qubits

### Simple Qubit
The most basic device is the simple qubit. This is simply defined from a qubit frequency given in GHz which defines the energy gap between $|0\rangle$ and  $|1\rangle$, along with the anharmonicity. If the anharmonicity is None, the Qubit is a simple two-level system, but with it defined, there will be a third level $|2\rangle$ with energy $2 f_{01} + \alpha$ where $\alpha$ is the anharmonicity.

Furthermore, the qubit can be defined with a decay by defining a $T_1 \neq 0$.
  
```python
SimpleQubit(
	frequency: float,
	anharmonicity: float = None,
	T1: float = 0.0,
):
```

The sweepable parameters of the SimpleQubit are:

| Parameter | Use                                      | Sweepable |
| --------- | ---------------------------------------- | --------- |
| frequency | The energy spacing between the 0 and 1 level in GHz | x         |
| anharmonicity    | The difference in energy splitting between 2-1 and 1-0 . (Given in Ghz)      |    x       |
| T1    | The characteristic time of qubit decay  | x         |

And the update methods calculates the following operators/dissipators:
- Hamiltonian
- Charge Matrix
- Dissipators
	- Qubit Decay


### Transmon 
The Transmon qubit defines an n-level anharmonic system from the physical parameters Transmon Device. The Hamiltonian and Charge Matrix are calculated numerically by diagonalizing the Hamiltonian in the charge basis of the charge matrix. (see [CircuitQ: an open-source toolbox for superconducting circuits](https://iopscience.iop.org/article/10.1088/1367-2630/ac8cab))

To define the Transmon call the following:
  
```python
SimpleQubit(
	self,
	EC: float,
	EJ: float,
	n_cutoff: int = 20,
	ng: float = 0.0,
	levels: int = 3,
	T1: float = 0.0,
)
```

The parameters of the Transmon qubit are:

| Parameter | Use                                                                 | Sweepable |
| --------- | ------------------------------------------------------------------- | --------- |
| EC        | Energy associated with capacitor in GHz                             | x         |
| EJ        | Energy assiciated with Josephson Junction in GHz                    | x         |
| n_cutoff  | Number of charges to include in calculations of hamiltnonian.       |           |
| ng        | Charge offset in units of 2e                                        | x         |
| levels    | Define how many energy levels of the Transmon should be considered. |           |
| T1        | The characteristic time of qubit decay process                      | x         |

And the update methods calculates the following operators/dissipators:
- Hamiltonian
- Charge Matrix
- Dissipators 
	- Qubit Decay

## Resonator
The resonator is a defined as a quantum harmonic oscillator with energy levels $(n + \frac12) 2 \pi f$ with $f$ the frequency of the resonator. It further supports decay of the resonator given by the characteristic time $\kappa$.

To define a resonator use the following:
  
```python
Resonator(
	frequency: float, 
	levels=10, 
	kappa: float = 0
)
```

The sweepable parameters of the Resonator are:

| Parameter | Use                                      | Sweepable |
| --------- | ---------------------------------------- | --------- |
| frequency | The energy spacing between levels in GHz | x         |
| levels    | The number of levels to consider         |           |
| kappa     | The characteristic time of photon decay  | x         |

Which updates the following operators and dissipators:
- Hamiltonian
- Coupling Operator
- Dissipators
	- Qubit Decay