import numpy as np
import streamlit as st
import scipy as smp 
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.optimize import fsolve 
import sympy as sp
import pandas as pd 
import warnings
%run Antoine_Coefficent_database.ipynb #Juypter notebook file of the database

T = (390) #kelvin
P = (5/1.013) # atm
#input name of components for extraction from the database
component_a = "N-PENTANE"
component_b = "N-HEXANE"
component_c = "CYCLOHEXANE"
component_d = "N-BUTANE"
component_e = "N-HEPTANE"
component_f = "N-OCTANE"
#mol of each component used for the VLE equilbrium calculations
mol_of_a = 0.50
mol_of_b = 0.30
mol_of_c = 0.20
mol_of_d = 0.0
mol_of_e = 0.0
mol_of_f = 0.0

total_mol = mol_of_a + mol_of_b + mol_of_c + mol_of_d + mol_of_e + mol_of_f

mol_fraction_a = mol_of_a / total_mol
mol_fraction_b = mol_of_b / total_mol
mol_fraction_c = mol_of_c / total_mol
mol_fraction_d = mol_of_d / total_mol
mol_fraction_e = mol_of_e / total_mol
mol_fraction_f = mol_of_f / total_mol


#obtaining saturation pressure for components at given Temperature for each component 
psat_a_atm = get_Psat_atm(component_a, T)
psat_b_atm = get_Psat_atm(component_b, T)
psat_c_atm = get_Psat_atm(component_c, T)
psat_d_atm = get_Psat_atm(component_d, T)
psat_e_atm = get_Psat_atm(component_e, T)
psat_f_atm = get_Psat_atm(component_f, T)



# find K values for each component 
K_a = psat_a_atm / P
K_b = psat_b_atm / P
K_c = psat_c_atm / P
K_d = psat_d_atm / P
K_e = psat_e_atm / P
K_f = psat_f_atm / P



#solve for the liquid fraction and subsequent X and Y compositions using rachford rice
def Rachford_Rice(L):
    x_a = mol_fraction_a / (((1 - L) * K_a ) + L)
    x_b = mol_fraction_b / (((1 - L) * K_b ) + L)
    x_c = mol_fraction_c / (((1 - L) * K_c ) + L)
    x_d = mol_fraction_d / (((1 - L) * K_d ) + L)
    x_e = mol_fraction_e / (((1 - L) * K_e ) + L)
    x_f = mol_fraction_f / (((1 - L) * K_f ) + L)
    return x_a + x_b + x_c + x_d + x_e + x_f - 1 

L_initial_estimate = 0.5 
L_solution = fsolve(Rachford_Rice, L_initial_estimate)[0]

if 0 <= L_solution <= 1:
    X_a = mol_fraction_a / (((1 - L_solution) * K_a ) + L_solution)
    X_b = mol_fraction_b / (((1 - L_solution) * K_b ) + L_solution)
    X_c = mol_fraction_c / (((1 - L_solution) * K_c ) + L_solution)
    X_d = mol_fraction_d / (((1 - L_solution) * K_d ) + L_solution)
    X_e = mol_fraction_e / (((1 - L_solution) * K_e ) + L_solution)
    X_f = mol_fraction_f / (((1 - L_solution) * K_f ) + L_solution)
    sum_x = X_a + X_b + X_c + X_d + X_e + X_f
    if abs(sum_x - 1) < 0.002:
        print('2 phase is present')
    else:
        print('no 2 phase')
    Y_a = X_a * K_a 
    Y_b = X_b * K_b 
    Y_c = X_c * K_c 
    Y_d = X_d * K_d
    Y_e = X_e * K_e 
    Y_f = X_f * K_f 
    Y = Y_a + Y_b + Y_c + Y_d + Y_e + Y_f
    V_solution = 1 - L_solution
    
    print('fractional liquid flowrate')
    print(L_solution)
    print('fractional vapour flowrate')
    print(V_solution)
    
    print('Sum of X')
    print(sum_x)
    
    print('Sum of Y')
    print(Y)
else:
    print('L solution out of bounds', L_solution)
    print("system may be single phase")

results = pd.DataFrame({
    "component": [component_a , component_b, component_c, component_d, component_e, component_f],
    "mol fraction": [mol_fraction_a, mol_fraction_b, mol_fraction_c, mol_fraction_d, mol_fraction_e, mol_fraction_f],
    "Psat (atm)": [psat_a_atm, psat_b_atm, psat_c_atm, psat_d_atm, psat_e_atm, psat_f_atm],
    "K value": [K_a, K_b, K_c, K_d, K_e, K_f],
    "x (Liquid Frac)": [X_a, X_b, X_c, X_d, X_e, X_f],
    "y (Vapor Frac)": [Y_a, Y_b, Y_c, Y_d, Y_e, Y_f]
})
print(results)
