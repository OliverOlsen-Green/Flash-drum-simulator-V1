# Flash-drum-simulator-V1
A python implementation for Vapor-Liquid (VLE) flash calculations for multicomponent systems using Antoine equations and the SRK equation of state.

Saturation pressure values are calcualated using Antoine Coefficent values obtained in the database have been sourced from NIST [https://www.nist.gov/].

## Features
Currently supported for up to 6 components that are available within the database present.

Uses a Rachford-Rice solver for phase equilibrium calculations 
The SRK flash calculation uses the phi-phi approach to determine the VLE of the mixture it also utilises the wilson correlation to make an initial estimate of the Ki values.
The SRK flash calculation for the mixture in the code differs slightly from values used within HYSES this is due to the assumption of ideal mixing since Kij parameters for the system are unknown.

## How to run
clone the repository:
``bash
   git clone https://github.com/OliverOlsen-Green/Flash-drum-simulator-V 
   
