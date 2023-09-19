In this note, we will compare the dipfferent methods for reading out either $\ket{0}$ or $\ket{1}$. The data is taken from the dataset described in: [[First Control of ADC Trace Data#The 1000 sample - 50 µs Data (large dataset)]].

The methods are tested and listed here:

```dataview

TABLE file.mtime as "Last Modified", AUC

FROM "Continuous Time Investigation/Weighting in Readout"

WHERE type = "method"

```

# Simple Filter

![[Simple Weights]]

# Matched Filter
![[Matched Filter]]

# Neural Network
![[Neural Network]]