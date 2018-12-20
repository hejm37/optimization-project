# optimization-project
Simple project for optimization Convex optimization and operations research 2018 Fall SYSU

## Problem A: Lasso

Run the command below in MATLAB command window to generate data required.
```
generate
```

Below algorithms are implemented:
* Proximal gradient method (ista.m)
* Alternating direction method (adm.m)
* Subgradient method (sgm.m)

Solve with different algorithm run
```
adm     % or ista, sgm
```

After iterations needed, result will show in figure window and command line window.


## Problem B: Logistic regression with MNIST

First download MNIST data and extract to 'problemB/MNIST/', then modify the parameter set in gd.m and run
```
gd
```
If weights are saved, you can use calculatiePlotResult.m to calculate the vectors needed for plotting then use the plotScript.m to plot the result.
