#todo 
- [ ] Create a Pulse Sequence, which simply adds multiple pulses in parallel or series

## Pulse Parent Class
At the moment the Pulse Parent class mostly serves a typing help. It just has single abstract method inheriting most functionality from the [Device Parent Class](Devices#Device Parent Class). A Pulse class takes the structure:

```python
class Pulse(Device):

    @abstractmethod
    def set_pulse(self):
	    A FUNCTION THAT DEFINES A PULSE AS
	    self.pulse: callable(t, args) -> float
```

> [!NOTE]
> With `np.piecewise` it should be possible to write this in a vectorized form. This could hopefully help with performance. 


## Square Cosine Pulse
The simplest pulse is the cosine pulse with a simple rectangular envelope. It is defined using the following:

```python
SquareCosinePulse(
	frequency: float,
	amplitude: float,
	start_time: float = 0,
	duration: float = None,
	phase: float = 0,
)
```

|Parameter|Function|Sweepable|
|----|----|----|
|`frequency`|Set the frequency of the pulse|x|
|`amplitude` |The amplitude|x|
|`start_time` |When the pulse starts|x|
|`duration` |How long it lasts|x|
|`phase` |A phase to give to the oscillating term|x|


## Gaussian Pulse
The simplest pulse is the square cosine pulse. It has the following arguments:

```python
GaussianPulse(
	frequency: float,
	amplitude: float,
	sigma: float,
	start_time=0,
	duration=0,
	phase=0,
	drag_alpha=0,
):
```

Where the parameters are given by the following:

|Parameter|Function|Sweepable|
|----|----|----|
|`frequency`|Set the frequency of the pulse|x|
|`amplitude` |The amplitude|x|
|`sigma` |The width of the pulse given as standard deviation of the gaussian envelope|x|
|`start_time` |When the pulse starts|x|
|`duration` |How long it lasts|x|
|`phase` |A phase to give to the oscillating term|x|
|`drag_alpha` |if DRAG should be applied to the pulse, this $\alpha_{DRAG}\neq0$ |x|
