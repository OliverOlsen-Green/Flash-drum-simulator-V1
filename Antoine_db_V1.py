import pandas as pd
import numpy as np
import streamlit as st
import scipy as smp 
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint
from scipy.optimize import fsolve 
import sympy as sp

#Antoine coefficents for each compound are sourced from NIST chemistry webbook. https://www.nist.gov/

antoine_data = pd.DataFrame([
    {"name": "METHANOL", "A": 5.15853, "B": 1569.613, "C": -34.846, "Tmin": 353.5, "Tmax": 512.63},
    {"name": "METHANOL", "A": 5.20409, "B": 1581.341, "C": -33.5, "Tmin": 288.1, "Tmax": 356.83},
    {"name": "METHANOL", "A": 5.31301, "B": 1676.569, "C": -21.728, "Tmin": 353, "Tmax": 483},
    {"name": "1-PROPANOL", "A": 4.57795, "B": 1221.423, "C": -87.474, "Tmin": 395.1, "Tmax": 508.24},
    {"name": "1-PROPANOL", "A": 4.861, "B": 1357.427, "C": -75.418, "Tmin": 329.92, "Tmax": 362.41},
    {"name": "1-PROPANOL", "A": 4.5968, "B": 1300.491, "C": -86.364, "Tmin": 405.46, "Tmax": 536.71},
    {"name": "ISO-BUTANOL", "A": 4.43126 , "B": 1236.991, "C": -101.528, "Tmin": 353.36, "Tmax": 388.77},
    {"name": "ISO-BUTANOL", "A":  4.40062, "B": 1260.453, "C": -92.588, "Tmin": 422.64, "Tmax": 547.71},
    {"name": "METHANE", "A": 3.9895, "B": 443.028, "C": -0.49, "Tmin": 90.99, "Tmax": 189.99},
    {"name": "METHANE", "A": 2.00253, "B": 125.819, "C": -48.823, "Tmin": 96.89, "Tmax": 110.19},
    {"name": "METHANE", "A": 3.80235, "B": 403.106, "C": -5.479, "Tmin": 93.04, "Tmax": 107.84},
    {"name": "METHANE", "A": 4.22061, "B": 516.689, "C": 11.223, "Tmin": 110.00, "Tmax": 190.5},
    {"name": "ETHANE", "A": 4.50706, "B": 791.2, "C": -6.422, "Tmin": 91.33, "Tmax": 144.13},
    {"name": "ETHANE", "A": 3.93835, "B": 659.739, "C": -16.719, "Tmin": 135.74, "Tmax": 199.91},
    {"name": "ETHENE", "A": 3.87261, "B": 584.146, "C": -18.307, "Tmin": 149.37, "Tmax": 188.57},
    {"name": "PROPANE", "A": 4.53678, "B": 1149.36, "C": 24.409, "Tmin": 277.6, "Tmax": 360.8},
    {"name": "PROPANE", "A": 3.98292, "B": 819.296, "C": -24.417, "Tmin": 230.6, "Tmax": 320.7},
    {"name": "PROPANE", "A": 4.01158, "B": 834.26, "C": -22.763, "Tmin": 166.02, "Tmax": 231.41},
    {"name": "N-BUTANE", "A": 4.70812, "B": 1200.475, "C": -13.013, "Tmin": 135.42, "Tmax": 212.89},
    {"name": "N-BUTANE", "A": 4.35576, "B": 1175.581, "C": -2.071, "Tmin": 272.66, "Tmax": 425},
    {"name": "N-BUTANE", "A": 3.85002, "B": 909.65, "C": -36.146, "Tmin": 195.11, "Tmax": 272.81},
    {"name": "ISO-BUTANE", "A": 4.3281, "B": 1132.108, "C": 0.918, "Tmin": 261.31, "Tmax": 408.12},
    {"name": "ISO-BUTANE", "A": 3.94417, "B": 912.141, "C": -29.908, "Tmin": 188.06, "Tmax": 261.54},
    {"name": "N-PENTANE", "A": 3.9892, "B": 1070.617, "C": -40.454, "Tmin": 268.8, "Tmax": 341.37},
    {"name": "N-HEPTANE", "A": 4.81803, "B": 1635.409, "C": -27.338, "Tmin": 185.29, "Tmax": 295.60},
    {"name": "N-HEPTANE", "A": 4.02832, "B": 1268.636, "C": -56.199, "Tmin": 299.07, "Tmax": 372.43},
    {"name": "N-OCTANE", "A": 5.2012, "B": 1936.281, "C": -20.143, "Tmin": 216.59, "Tmax": 297.10},
    {"name": "ETHANE", "A": 4.04867, "B": 1355.126, "C": -63.633, "Tmin": 326.08, "Tmax": 399.72},
    {"name": "N-HEXANE", "A": 3.45604, "B": 1044.038, "C": -53.893, "Tmin": 177.70, "Tmax": 264.32},
    {"name": "N-HEXANE", "A": 4.00266, "B": 1171.53, "C": -48.784, "Tmin": 286.18, "Tmax": 342.69},
    {"name": "CYCLOHEXANE", "A": 4.13983, "B": 1316.554, "C": -35.581, "Tmin": 323.0, "Tmax": 523},
    {"name": "CYCLOHEXANE", "A": 3.9920, "B": 1216.93, "C": -48.621, "Tmin": 303.0, "Tmax": 343},
    {"name": "CYCLOHEXANE", "A": 3.17125, "B": 780.637, "C": -107.29, "Tmin": 315.70, "Tmax": 353.90},
    {"name": "CYCLOHEXANE", "A": 3.96988, "B": 1203.526, "C": -50.287, "Tmin": 293.06, "Tmax": 354.73},])
    
    
    

def get_Psat_atm(component, T):
    # check if the component exists in the database
    component_data = antoine_data[antoine_data["name"] == component]
    if component_data.empty:
        raise ValueError(f"Component {component} not found in the database")
    
    # find row with temperature in range
    row = component_data[
        (component_data["Tmin"] <= T) &
        (component_data["Tmax"] >= T)
    ]
    
    if row.empty:
        # if no exact match, find the closest temperature range
        print(f"Warning: No Antoine coefficients found for {component} at {T} K")
        print(f"Available temperature ranges for {component}:")
        for _, r in component_data.iterrows():
            print(f"  {r['Tmin']} K to {r['Tmax']} K")
        
        # cse the closest range (this is a fallback solution)
        if T_K < component_data["Tmin"].min():
            row = component_data[component_data["Tmin"] == component_data["Tmin"].min()]
            print(f"Using coefficients for the lowest available range: {row['Tmin'].values[0]} K to {row['Tmax'].values[0]} K")
        else:
            row = component_data[component_data["Tmax"] == component_data["Tmax"].max()]
            print(f"Using coefficients for the highest available range: {row['Tmin'].values[0]} K to {row['Tmax'].values[0]} K")

    
    A = row["A"].values[0]
    B = row["B"].values[0]
    C = row["C"].values[0]
    
    # calculate saturation pressure - corrected Antoine equation
    Psat_bar = 10 ** (A - B / (T + C))
    Psat_atm = Psat_bar / 1.013
    return Psat_atm
