# Orthogonal Fitting using emcee

[This file](OrthogonalFitting.py) contains the code that executes the MCMC sampler (emcee) using an orthogonal likelihood. The code accepts two arrays for the x and y variables with errors on both variables. A [test data file](data.py) is included to show an example code run.

## Input
csv file with four columns (x, err_x, y, err_y) of data that needs to be fit to a line

## Output
.pkl file with the MCMC flat samples
best fit parameters (slope, intercept and intrinsic scatter)
total scatter (orthogonal and vertical)
best fit and corner plots from the MCMC run

