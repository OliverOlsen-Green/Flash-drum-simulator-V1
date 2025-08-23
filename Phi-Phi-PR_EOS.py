!pip install thermo
from thermo import chemical 
from scipy.optimize import fsolve
import numpy as np
import streamlit as st
import scipy as smp 
import sympy as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
from scipy.optimize import brentq
R = 83.14 #abr.cm/mol.K
T = (-80 + 273.15) # K
P = 20 # bar
supercritical_data = np.array([
    [190.70, 46.4068, 0.0115],   # Methane
    [305.43, 48.8385, 0.0986],   # Ethane
    [369.90, 42.5666, 0.1524],   # Propane
    [408.10, 36.4762, 0.18479],  # Isobutane
    [425.20, 37.9662, 0.2010],   # n-Butane
    [126.19, 33.9437, 0.0400]    # Nitrogen
])
component_names = ["methane","ethane","propane","isobutane","n-butane","nitrogen"]
mol_fraction = [0.7812, 0.03, 0.02, 0.0048, 0.001, 0.163]
kij_matrix = [
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # iso-propanol row
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # methanol row
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],   
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
]

MAX_ITER = 100
TOL = 1e-4
def Wilson_Correlation(P, T, R, Tc, Pc, acentric_factor):
    Pr = P / Pc
    Tr = T / Tc
    Ki = (math.exp(5.37 * (1 + acentric_factor) * (1 - ( 1 / Tr)))) / Pr
    return Ki

K_values = []
for i, data in enumerate(supercritical_data):
    Tc = data[0]
    Pc = data[1]
    acentric_factor = data[2]
    K_i = Wilson_Correlation(P, T, R, Tc, Pc, acentric_factor)
    K_values.append({
        "Component": component_names[i],
        "K values": K_i
    })

dk = pd.DataFrame(K_values)
print(dk)   

z = np.array(mol_fraction, dtype=float)
K = np.array([row["K values"] for row in K_values], dtype=float)

# Rachford–Rice solver
def rachford_rice(L):
    
    eps = 1e-12
    L = float(np.clip(L, eps, 1.0 - eps))
    denom = (1.0 - L) * K + L
    return np.sum(z / denom) - 1.0
X_1 = np.sum(z / K)            
V_1 = np.sum(z * K)  


g0  = X_1 - 1.0               
g1 = rachford_rice(1.0 - 1e-12)   

if g0 * g1 < 0.0:
    L_solution = brentq(rachford_rice, 1e-12, 1.0 - 1e-12,
                        xtol=1e-12, rtol=1e-10, maxiter=200)
else:
    # decide phase properly using both sums
    if (X_1 < 1.0) and (V_1 < 1.0):
        L_solution = 1.0        # all liquid
    elif (X_1 > 1.0) and (V_1 > 1.0):
        L_solution = 0.0        # all vapor


den = (1.0 - L_solution) * K + L_solution
x = z / den
y = K * x
ysum = y.sum()
if ysum > 0.0:
    y = y / ysum

# Prepare results for display
results_flash = []
for i, comp in enumerate(component_names[:6]):
    results_flash.append({
        "Component": comp,
        "Feed (z)": mol_fraction[i],
        "K-value": K_values[i],
        "Liquid (x)": x[i],
        "Vapor (y)": y[i]
    })

# Display results
dt = pd.DataFrame(results_flash)
print("\nFlash calculation results:")
print(dt)
print(f"Liquid fraction (L): {L_solution}")
Z = sp.Symbol('Z')

def PR_EOS(Z, Tc, Pc, acentric_factor):
    a = 0.45724 * ((R ** 2) * (Tc ** 2)) / Pc
    b = 0.07780 * R * Tc / Pc

    if acentric_factor < 0.49:
        m = 0.3796 + 1.54226 * acentric_factor - 0.2699 * acentric_factor**2
    else:
        m = (0.379642 + 1.48503 * acentric_factor - 0.1644 * acentric_factor**2 + 0.016667 * acentric_factor**3)

    alpha = (1 + m * (1 - math.sqrt(T / Tc))) ** 2
    a_alpha = a * alpha
    A = (a_alpha * P) / ((R * T) ** 2)
    B = (b * P) / (R * T)

    PR_eq = Z**3 + (B - 1)*Z**2 + (A - 3*B**2 - 2*B)*Z - (A*B - B**2 - B**3)
    return PR_eq, a, b, alpha, a_alpha, A, B

results = []

for i, data in enumerate(supercritical_data):
    Tc, Pc, ω = data
    PR_eq, a, b, alpha, a_alpha, A, B = PR_EOS(Z, Tc, Pc, ω)
    results.append({
        "Component": component_names[i],
        "a (bar·cm⁶/mol²)": round(a, 4),
        "b (cm³/mol)": round(b, 4),
        "A": round(A, 4),
        "B": round(B, 4),
        "alpha": round(alpha, 4),
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
a_list = df["a (bar·cm⁶/mol²)"].tolist()
b_list = df["b (cm³/mol)"].tolist()
x = [row["Liquid (x)"] for row in results_flash]
y = [row["Vapor (y)"] for row in results_flash]
alpha_list = df[("alpha")].tolist()


def PR_mixture_parameters_liquid(a_list, b_list, x, alpha_list):
    a_mix_liquid = 0.0
    b_mix_liquid = 0.0  # Initialize outside the loops
    
    for i in range(len(x)):
        b_mix_liquid += x[i] * b_list[i] 
        for j in range(len(x)):
            a_i = a_list[i] * alpha_list[i] 
            a_j = a_list[j] * alpha_list[j]  
            kij = kij_matrix[i][j]
           
            a_mix_liquid += x[i] * x[j] * math.sqrt(a_i * a_j) * (1 - kij)
            
    return a_mix_liquid, b_mix_liquid
def PR_mixture_parameters_vapour(a_list, b_list, y, alpha_list):
    a_mix_vapour = 0.0
    b_mix_vapour = 0.0  # Initialize outside the loops
    
    for i in range(len(y)):
        b_mix_vapour += y[i] * b_list[i]  
        for j in range(len(y)):
            a_i = a_list[i] * alpha_list[i]  
            a_j = a_list[j] * alpha_list[j]  
            kij = kij_matrix[i][j]
            
            a_mix_vapour += y[i] * y[j] * math.sqrt(a_i * a_j) * (1 - kij)
            
    return a_mix_vapour, b_mix_vapour
a_mix_liquid, b_mix_liquid = PR_mixture_parameters_liquid(a_list, b_list, x, alpha_list)
print("\nMixture values using van der Waals mixing rule (kij = 0):")
print(f"Mixed a liquid (bar·cm⁶/mol²): {round(a_mix_liquid, 4)}")
print(f"Mixed b liquid (cm³/mol): {round(b_mix_liquid, 4)}")
a_mix_vapour, b_mix_vapour = PR_mixture_parameters_vapour(a_list, b_list, y, alpha_list)
print("\nMixture values using van der Waals mixing rule (kij = 0):")
print(f"Mixed a vapour (bar·cm⁶/mol²): {round(a_mix_vapour, 4)}")
print(f"Mixed b vapour (cm³/mol): {round(b_mix_vapour, 4)}")
Z_L = sp.Symbol('Z_l')
Z_V = sp.Symbol('Z_V')

def PR_EOS_Mix_liquid(Z_L):
    A_l = (a_mix_liquid * P) / ((R ** 2) * ( T ** 2))
    B_l = (b_mix_liquid * P) / (R * T)
    PR_EOS_liquid_mixture = (Z_L ** 3) + ((B_l - 1) * (Z_L ** 2)) + ((A_l - (3 * (B_l ** 2)) - (2 * B_l)) * Z_L) - ((A_l * B_l) - (B_l ** 2) - (B_l ** 3))
    return PR_EOS_liquid_mixture, A_l, B_l

PR_EOS_liquid_mixture, A_l, B_l = PR_EOS_Mix_liquid(Z_L)
eq_liquid = sp.N(PR_EOS_liquid_mixture)

z_l_roots = sp.solve(eq_liquid, Z_L)
z_l =  min(z.as_real_imag()[0] for z in z_l_roots
           if abs(z.as_real_imag()[1]) < 1e-6
)
print("compressibilty factor of the liquid mixture")
print(z_l)
def PR_EOS_Mix_Vapour(Z_V):
    A_V = (a_mix_vapour * P) / ((R ** 2) * ( T ** 2))
    B_V = (b_mix_vapour * P) / (R * T)
    PR_EOS_vapour_mixture = (Z_V ** 3) + ((B_V - 1) * (Z_V ** 2)) + ((A_V - (3 * (B_V ** 2)) - (2 * B_V)) * Z_V) - ((A_V * B_V) - (B_V ** 2) - (B_V ** 3))
    return PR_EOS_vapour_mixture, A_V, B_V

PR_EOS_vapour_mixture, A_V, B_V = PR_EOS_Mix_Vapour(Z_V)
eq_vapour = sp.N(PR_EOS_vapour_mixture)

z_v_roots = sp.solve(eq_vapour, Z_V)
z_v =  max(z.as_real_imag()[0] for z in z_v_roots
           if abs(z.as_real_imag()[1]) < 1e-6
)
print("compressibility factor of the vapour phase mixture")
print(z_v)

print(A_l)
print(B_l)
print(A_V)
print(B_V)
def AA_i_liquid(a_list, alpha_list, x, a_mix_liquid):
    n = len(a_list)
    AA_i_liquid_list = []
    for i in range(n):
        sum_aa_ij_liquid = 0
        for j in range(n):
            a_alpha_ij_liquid = math.sqrt(a_list[i] * alpha_list[i] * a_list[j] * alpha_list[j])
            sum_aa_ij_liquid += x[j] * a_alpha_ij_liquid
        AA_i_L = ((2 * sum_aa_ij_liquid) / a_mix_liquid) 
        AA_i_liquid_list.append(AA_i_L)
    return AA_i_liquid_list

def AA_i_vapour(a_list, alpha_list, y, a_mix_vapour):
    n = len(a_list)
    AA_i_vapour_list = []
    for i in range(n):
        sum_aa_ij_vapour = 0
        for j in range(n):
            a_alpha_ij_vapour = math.sqrt(a_list[i] * alpha_list[i] * a_list[j] * alpha_list[j])
            sum_aa_ij_vapour += y[j] * a_alpha_ij_vapour
        AA_i_V = ((2 * sum_aa_ij_vapour) / a_mix_vapour) 
        AA_i_vapour_list.append(AA_i_V)
    return AA_i_vapour_list
AA_i_l = AA_i_liquid(a_list, alpha_list, x, a_mix_liquid)
AA_i_v = AA_i_vapour(a_list, alpha_list, y, a_mix_vapour)

print(component_names)
print("AA values of each component in the liquid phase")
print(AA_i_l)
print("AA values of each compoonent in the vapour phase")
print(AA_i_v)
def fugacity_PR_EOS(a_list, b_list, AA_i_list, Z, A, B, b_mix, phase_label):
    phi_list = []
    component_names = ['iso-propanol', 'methanol', 'iso-butanol']
    
    sqrt2 = math.sqrt(2)
    
    for i in range(len(b_list)):
        B_i = b_list[i]
        A_i = a_list[i]
        AA_i = AA_i_list[i]
        term1 = (B_i * (Z - 1)) / b_mix
        term2 = math.log(max(Z - B, 1e-10))  
        
        
        numerator = Z + (1 + sqrt2) * B
        denominator = Z + (1 - sqrt2) * B
        
        # make sure both numerator and denominator are positive
        if numerator <= 0 or denominator <= 0:
            
            term3 = 0  # Or some appropriate default value
        else:
            term3 = ((A / (2 * sqrt2 * B)) * (AA_i - (B_i / b_mix)) * 
                    math.log(numerator / denominator))
        
        ln_phi = term1 - term2 - term3
        phi = math.exp(ln_phi)
        phi_list.append(phi)
    
    # Return the entire list, not just the first value
    return phi_list



       


phi_l = fugacity_PR_EOS(
    a_list = a_list,
    b_list = b_list,
    AA_i_list = AA_i_l,
    Z = z_l,
    A = A_l,
    B = B_l,
    b_mix = b_mix_liquid,
    phase_label = "liquid"
)

phi_v = fugacity_PR_EOS(
    a_list = a_list,
    b_list = b_list,
    AA_i_list = AA_i_v,
    Z = z_v,
    A = A_V,
    B = B_V,
    b_mix = b_mix_vapour,
    phase_label = "vapour"
)
print('liquid fugacity coefficents')
print(phi_l)
print('vapour fugacity coefficents')
print(phi_v)
K = [phi_l[i] / phi_v[i] for i in range(len(phi_l))]
print("\nK values:")
for i, k_val in enumerate(K):
    print(f"{component_names[i]}: {k_val:.6f}")

print(f"A_liquid: {A_l}, B_liquid: {B_l}, Z_l: {z_l}")
print(f"A_vapour: {A_V}, B_vapour: {B_V}, Z_v: {z_v}")


K_val = K.copy()
z = mol_fraction.copy()


for iteration in range(MAX_ITER):
    # old K stored for convergence check
    K_old = K_val.copy()
    Z_L = sp.Symbol('Z_L')
    Z_V = sp.Symbol('Z_V')
    
   
    def Rachford_Rice(L):
        x_values = [z[i] / (((1 - L) * K_val[i]) + L) for i in range(len(z))]
        return sum(x_values) - 1
        

    L_initial_estimate = L_solution
    L_solution = fsolve(Rachford_Rice, L_initial_estimate)[0]

    
    x = [z[i] / (((1 - L_solution)* K_val[i]) + L_solution ) for i in range(len(z))]
    y = [x[i] * K_val[i] for i in range(len(z))]
    sy = sum(y)
    if sy > 0:
        y = [yi/sy for yi in y]
   
    a_list = []
    b_list = []
    alpha_list_liquid = []
    a_list_vapour = []  # Make sure these are defined as lists
    b_list_vapour = []
    alpha_list_vapour = []

    for i in range(len(supercritical_data)):
        Tc, Pc, omega = supercritical_data[i]
        PR_eq_liq, a, b, alpha, a_alpha, A, B = PR_EOS(Z, Tc, Pc, omega)  # Make sure PR_EOS is a function, not a list
        a_list.append(a)
        b_list.append(b)
        alpha_list_liquid.append(alpha)
        a_list_vapour.append(a)
        b_list_vapour.append(b)
        alpha_list_vapour.append(alpha)

    #mixture values for the liquid and vapour phases
    a_mix_liquid, b_mix_liquid = PR_mixture_parameters_liquid(a_list, b_list, x, alpha_list_liquid)
    a_mix_vapour, b_mix_vapour = PR_mixture_parameters_vapour(a_list_vapour, b_list_vapour, y, alpha_list_vapour)

    PR_EOS_liquid_mixture, A_l, B_l = PR_EOS_Mix_liquid(Z_L)
    z_l_roots = sp.solve(sp.N(PR_EOS_liquid_mixture), Z_L)
    # Rename the loop variable to avoid conflict with the z list
    z_l = min(root.as_real_imag()[0] for root in z_l_roots if abs(root.as_real_imag()[1]) < 1e-6)

    PR_EOS_vapour_mixture, A_V, B_V = PR_EOS_Mix_Vapour(Z_V)
    z_v_roots = sp.solve(sp.N(PR_EOS_vapour_mixture), Z_V)
    # Rename the loop variable to avoid conflict with the z list
    z_v = max(root.as_real_imag()[0] for root in z_v_roots if abs(root.as_real_imag()[1]) < 1e-6)

    #AA_i values for the liquid and vapour phase
    AA_i_L = AA_i_liquid(a_list, alpha_list_liquid, x, a_mix_liquid)
    AA_i_V = AA_i_vapour(a_list_vapour, alpha_list_vapour, y, a_mix_vapour)

    
  
    phi_l = fugacity_PR_EOS(a_list, b_list, AA_i_L, z_l, A_l, B_l, b_mix_liquid, "liquid")
    phi_v = fugacity_PR_EOS(a_list_vapour, b_list_vapour, AA_i_V, z_v, A_V, B_V, b_mix_vapour, "vapour")

    
    new_K_values = [phi_l[i] / phi_v[i] for i in range(len(phi_l))]
    max_diff = max(abs((new_K_values[i] - K_old[i]) / K_old[i]) for i in range(len(K_val)))
    K_val = new_K_values.copy()
    L_initial_estimate = L_solution
    print(f"Iter {iteration+1} | Max K diff: {max_diff:.6f}")
    if max_diff < TOL:
        sum_x = sum(x)
        sum_y = sum(y)
        x_tol = abs(sum_x - 1) < 0.02
        y_tol = abs(sum_y - 1) < 0.02
        if x_tol and y_tol:
            print("\nConverged!")
            print(f"Final Liquid fraction (L): {L_solution:.6f}")
            print(f"Sum of x: {sum_x:.6f} | Sum of y: {sum_y:.6f}")
            print("\nComponent   x           y           K")
            for i, name in enumerate(component_names):
                print(f"{name:12} {x[i]:.6f}   {y[i]:.6f}   {new_K_values[i]:.6f}")
            break
        
else:
    print("\nDid not converge after maximum iteration")
