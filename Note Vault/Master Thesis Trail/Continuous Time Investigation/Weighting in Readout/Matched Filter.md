---
AUC: "0.8911"
type: method
---

This is a commonly used method, where the points in time is scaled such that the most separated values in time between the ground- and excited state are weighted the highest.

Filter:
![[Matched Filter-2.png]]

Separation (almost  identical)
![[Matched Filter.png]]

Evaluation
![[Matched Filter-1.png]]




## Longer Duration
Better for long readouts where we hit the $T_{1}$ effects. See pulse of $25 \text{ µs}$
![[Matched Filter-3.png]]

with these weights. So it quickly locks in:
![[Matched Filter-5.png]]

Compared with simple weights which yields the following:
![[Matched Filter-4.png]]