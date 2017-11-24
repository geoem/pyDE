# PyDE #

A python implementation of the differential evolution metaheuristic search algorithm introduced by R. Storn and K. Price to solve constrained optimization problems with continuous design variables.

https://doi.org/10.1023/A:1008202821328

### Prerequisites ###

* Python 2.7
* Numpy >= 1.9.2

### Usage ###

```
python pyde.py
```

### Comments ###
1. Only DE/rand/1/bin scheme is implemented. User can efficiently add more de schemes in the mutation method of DifferentialEvolution class.
2. The problems are taken from: J.J. Liang, T.P. Runarsson, E. Mezura-Montes, M. Clerc, P.N. Suganthan, C.A. Coello Coello, K. Deb. 
   Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization, Technical Report, 2006. 
   http://www.ntu.edu.sg/home/EPNSugan

### Author ###

    Manolis Georgioudakis (geoem@mail.ntua.gr)

Copyright: 2017