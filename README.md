# mnprobitstan
multinomial probit regression in stan / cmdstanr

This code generates 500 observations of one of 7 different dichotomous outcomes predicted by time, clustered by journal and field.

It then fits a stan model of a multinomial multilevel probit model that allows for residual correlation.

It also demonstrates some plotting options via conversion to an rstan object, and using ggplot2.

There is a LOT of room for improvement in the code, and in the prior specification. But this works as a proof of concept.
