---
date: 2023-07-20
code: /mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/projects/in_measurement_calibration/control_of_test_data.py; /mnt/c/Users/johan/OneDrive/Skrivebord/Master_Thesis/projects/in_measurement_calibration/control_of_longer_data.py
---

## Plots of Different Investigations

### The 100 sample - 5 µs Data (small dataset)

We demodulate and plot the trajectories. We take the first 5 µs and the 1 µs and plot the trajectories. We both do the total cumulative and a cumulative mean. The cumulative mean has 100 ns warm-up, so we don't plot the extreme noise which is seen in the beginning when dividing with low numbers. On these plots the mean path is also displayed with alpha = 1.0.

![[test_data_trajectories.png]]

### The 1000 sample - 50 µs Data (large dataset)

This sample is demodulated and course grained $\to$ We mean over every $10 ns$ to reduce noise and significantly reduce the size of the data.

![[longer_data_trajectories.png]]