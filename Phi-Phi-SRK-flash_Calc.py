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
from math import pi

R = 83.14 #bar.cm/mol.K
T = 400 #kelvin 
P = 4 / 1.013 #bar
Z_flowrate = 100 #kmol/hr
#supercritical data (Pressure (bar), temperature (K) and acentric factor)
supercritical_data = np.array([
    [508.3, 47.6, 0.665],
    [512.6, 80.9, 0.556],
    [547.8, 43.0, 0.592]
])
kij_matrix = [
    [0.0, 0.0, 0.0],   # iso-propanol row
    [0.0, 0.0, 0.0],   # methanol row
    [0.0, 0.0, 0.0]    # i-Butanol row
]
MAX_ITER = 120
TOL = 1e-4
component_names = ["iso-propanol","methanol","iso-butanol"]
Mr = np.array([60.01, 32.04, 74.122]) # molar mass (g/mol)
mol_fraction = np.array([1/3, 1/3, 1/3]) 
def wilson_correlation(P, T, Tc, Pc, accentric_factor):
    Pr = P / Pc
    Tr = T / Tc
    Ki = (math.exp(5.37*(1 + accentric_factor) * (1 - ( 1 / Tr)))) / Pr
    return Ki
K_values = []
K_values_display = []
for i, data in enumerate(supercritical_data):
    Tc = data[0]
    Pc = data[1]
    accentric_factor = data[2]
    K_i = wilson_correlation(P, T, Tc, Pc, accentric_factor)
    K_values_display.append({
        "component": component_names[i],
        "K values": K_i
    })
   
dk = pd.DataFrame(K_values_display)
print(dk)

z = np.array(mol_fraction, dtype=float)
K = np.array([row["K values"] for row in K_values_display], dtype=float)

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


results_flash = []
for i, comp in enumerate(component_names[:3]): 
    results_flash.append({
        "Component": comp,
        "Feed (z)": mol_fraction[i],
        "K-value": K[i],
        "Liquid (x)": x[i],
        "Vapor (y)": y[i]
    })

# display results of the flash calculation 
dt = pd.DataFrame(results_flash)
print("\nFlash calculation results:")
print(dt)
print(f"Liquid fraction (L): {L_solution}")

Z = sp.Symbol('Z')
def SRK(Z, Tc, Pc, accentric_factor):
    a = 0.427480 * (((R ** 2) * (Tc ** 2))/ Pc)
    b = 0.08664 * ((Tc * R )/ Pc)
    m = 0.48508 + (1.55171 * accentric_factor) - (0.15613 * (accentric_factor ** 2)) 
    alpha = (1 + (m * (1 - math.sqrt(T / Tc)))) ** 2
    alpha_a = alpha * a
    A = (alpha_a * P)/((R * T) ** 2) 
    B = (b * P) /(R * T)
    SRK_eq = (((Z ** 3) - ((Z ** 2)) + ((A - B - (B ** 2))) * Z) - (A * B))
    return SRK_eq , A , B, a, b, alpha

results = []

for i , data in enumerate(supercritical_data):
    Tc = data[0]
    Pc = data[1]
    accentric_factor = data[2]
    
    SRK_eq, A, B, a, b, alpha = SRK(Z, Tc, Pc, accentric_factor)
    results.append({
        "Component": component_names[i],
        "a (bar·cm⁶/mol²)": round(a, 4),
        "b (cm³/mol)": round(b, 4),
        "A": round(A,4),
        "B": round(B,4),
        "alpha": round(alpha, 4),
    })

df = pd.DataFrame(results)
print(df.to_string(index=False))

a_list = df["a (bar·cm⁶/mol²)"].tolist()
b_list = df["b (cm³/mol)"].tolist()
x = [row["Liquid (x)"] for row in results_flash]
y = [row["Vapor (y)"] for row in results_flash]
alpha_list = df[("alpha")].tolist()

# a and b for the liquid phase 
def SRK_mixture_parameters_liquid(a_list, b_list, x, alpha_list):
    a_mix_liquid = 0.0
    b_mix_liquid = 0.0  
    
    for i in range(len(x)):
        b_mix_liquid += x[i] * b_list[i]  
        for j in range(len(x)):
            a_i = a_list[i] * alpha_list[i] 
            a_j = a_list[j] * alpha_list[j]  
            kij = kij_matrix[i][j]
            a_mix_liquid += x[i] * x[j] * math.sqrt(a_i * a_j) * (1 - kij)
            
    return a_mix_liquid, b_mix_liquid


# a and b of the mixture for the vapour phase
def SRK_mixture_parameters_vapour(a_list, b_list, y, alpha_list):
    a_mix_vapour = 0.0
    b_mix_vapour = 0.0  
    
    for i in range(len(y)):
        b_mix_vapour += y[i] * b_list[i] 
        for j in range(len(y)):
            a_i = a_list[i] * alpha_list[i]  
            a_j = a_list[j] * alpha_list[j]  
            kij = kij_matrix[i][j]
            a_mix_vapour += y[i] * y[j] * math.sqrt(a_i * a_j) * (1 - kij)
            
    return a_mix_vapour, b_mix_vapour
a_mix_liquid, b_mix_liquid = SRK_mixture_parameters_liquid(a_list, b_list, x, alpha_list)
print("\nMixture values using van der Waals mixing rule (kij = 0):")
print(f"Mixed a liquid (bar·cm⁶/mol²): {round(a_mix_liquid, 4)}")
print(f"Mixed b liquid (cm³/mol): {round(b_mix_liquid, 4)}")
a_mix_vapour, b_mix_vapour = SRK_mixture_parameters_vapour(a_list, b_list, y, alpha_list)
print("\nMixture values using van der Waals mixing rule (kij = 0):")
print(f"Mixed a vapour (bar·cm⁶/mol²): {round(a_mix_vapour, 4)}")
print(f"Mixed b vapour (cm³/mol): {round(b_mix_vapour, 4)}")

Z_val = sp.Symbol('Z_val')
Z_Val = sp.Symbol('Z_Val')
#SRK EOS for the liquid phase mixture
def SRK_Mixture_liquid(Z_val):
    A_liquid = (a_mix_liquid * P)/((R * T) ** 2) 
    B_liquid = (b_mix_liquid * P)/(R * T)
    SRK_liquid = (Z_val**3) - (Z_val**2) + (A_liquid - B_liquid - (B_liquid**2))*(Z_val) - (A_liquid*B_liquid)
    return SRK_liquid , A_liquid, B_liquid

# SRK EOS for the vapour mixture
def SRK_Mixture_vapor(Z_Val):
    A_vapour = (a_mix_vapour * P)/((R * T) ** 2)
    B_vapour = (b_mix_vapour * P)/(R * T)
    SRK_vapour = (Z_Val**3) - (Z_Val**2) + (A_vapour - B_vapour - (B_vapour**2))*(Z_Val) - (A_vapour*B_vapour)
    return SRK_vapour , A_vapour, B_vapour

SRK_liquid,  A_liquid, B_liquid = SRK_Mixture_liquid(Z_val)
eq_liquid = sp.N(SRK_liquid)
SRK_vapour, A_vapour, B_vapour = SRK_Mixture_vapor(Z_Val)
eq_vapour = sp.N(SRK_vapour)
# roots for the liquid phase mixture
Z_l_roots = sp.solve(eq_liquid, Z_val)

# roots for the vapour phase mixture 
Z_v_roots = sp.solve(eq_vapour, Z_Val)
# smallest root for the liquid phase
Z_l =  min(
    z.as_real_imag()[0] for z in Z_l_roots
    if abs(z.as_real_imag()[1]) < 1e-6
)

# max root for compressiblity of the vapour phase
Z_v = max(
    z.as_real_imag()[0] for z in Z_v_roots
    if abs(z.as_real_imag()[1]) < 1e-6
)

print(f"Liquid phase compressibility factor (Z): {Z_l:.6f}")
print(f"Vapor phase compressibility factor (Z): {Z_v:.6f}")
print(A_liquid)
print(A_vapour)
print(B_liquid)
print(B_vapour)

print("a_list (liquid):", a_list)
print("a_list_vapour:", a_list)
print("x:", x)
print("y:", y)
# AAi for the liquid phase
def calculate_AA_i(a_list, alpha_list, a_mix_liquid, x):
    n = len(a_list)
    AA_i_list = []

    for i in range(n):
        sum_aa_ij = 0.0
        for j in range(n):
            a_i_alpha = a_list[i] * alpha_list[i]
            a_j_alpha = a_list[j] * alpha_list[j]
            a_alpha_ij = math.sqrt(a_i_alpha * a_j_alpha)# kij = 0
            sum_aa_ij += x[j] * a_alpha_ij
        AA_i = ((2 ) / a_mix_liquid) * sum_aa_ij
        AA_i_list.append(AA_i)

    return AA_i_list

# for the vapour phase
def calculate_AA_i_vapour(a_list, alpha_list, y, a_mix_vapour):
    n = len(a_list)
    AA_i_vapour_list = []

    for i in range(n):
        sum_aa_ij_vapour = 0.0
        for j in range(n):
            a_i_alpha_vapour = a_list[i] * alpha_list[i]
            a_j_alpha_vapour = a_list[j] * alpha_list[j]
            a_alpha_ij_vapour = math.sqrt(a_i_alpha_vapour * a_j_alpha_vapour)# kij = 0
            sum_aa_ij_vapour += y[j] * a_alpha_ij_vapour
        AA_i_vapour = ((2 ) / a_mix_vapour) * sum_aa_ij_vapour
        AA_i_vapour_list.append(AA_i_vapour)

    return AA_i_vapour_list

AA_i_liquid = calculate_AA_i(a_list, alpha_list, a_mix_liquid, x)
AA_i_vapour = calculate_AA_i_vapour(a_list, alpha_list, y, a_mix_vapour)
print(AA_i_liquid)
print(AA_i_vapour)

def fugacity_srk(b_list, a_list, AA_i_list, Z, A, B, b_mix, phase_label):
    phi_list = []
    component_names = ['2-Propanol', 'Methanol', 'i-Butanol']

    for i in range(len(b_list)):
        B_i = b_list[i]
        A_i = a_list[i]
        AA_i = AA_i_list[i]

       
        ln_phi = (((B_i)*(Z-1)) / b_mix) - math.log(Z-B) - ((A/B)*((AA_i)-(B_i/b_mix))*math.log(1+(B/Z)))
        
        phi = math.exp(ln_phi)
        phi_list.append(phi)
        
        

    return phi_list


# fugacity for the liquid phase
phi_l = fugacity_srk(
    b_list=b_list,               # from: results[Phase=="Liquid"]
    a_list=a_list,
    AA_i_list=AA_i_liquid,
    Z=Z_l,
    A=A_liquid,
    B=B_liquid,
    b_mix=b_mix_liquid,
    phase_label="liquid"
)

# fugacity for vapor phase
phi_v = fugacity_srk(
    b_list=b_list,
    a_list=a_list,
    AA_i_list=AA_i_vapour,
    Z=Z_v,
    A=A_vapour,
    B=B_vapour,
    b_mix=b_mix_vapour,
    phase_label="vapour"
)
# calculate K values 
K = [phi_l[i] / phi_v[i] for i in range(len(phi_l))]
print(phi_l)
print(phi_v)
# Print K values
print("\nK values:")
for i, k_val in enumerate(K):
    print(f"{component_names[i]}: {k_val:.6f}")


K_val = K.copy()                
z = mol_fraction.copy()        # feed mole fractions    

for iteration in range(MAX_ITER):
    # save previous K-values for convergence check
    K_old = K_val.copy()
    Z_L = sp.Symbol('Z_L')
    Z_V = sp.Symbol('Z_V')

    # Rachford-Rice with phi-phi K-values
    def Rachford_Rice(L):
        x_values = [z[i] / (((1 - L) * K_val[i]) + L) for i in range(len(z))]
        return sum(x_values) - 1
    L_initial_estimate = L_solution
    L_solution = fsolve(Rachford_Rice, L_initial_estimate)[0]

    # compute x and y with new L
    x = [z[i] / (((1 - L_solution)* K_val[i]) + L_solution ) for i in range(len(z))]
    y = [x[i] * K_val[i] for i in range(len(z))]
    sy = sum(y)
    if sy > 0:
        y = [yi/sy for yi in y]

    # pure-component parameters from SRK
    a_list = []
    b_list = []
    alpha_list = []

    for i in range(len(supercritical_data)):
        Tc, Pc, omega = supercritical_data[i]
        SRK_eq, A, B, a, b, alpha = SRK(Z, Tc, Pc, omega)
        a_list.append(a)
        b_list.append(b)
        alpha_list.append(alpha)

    # mixture parameters 
    a_mix_l, b_mix_l = SRK_mixture_parameters_liquid(a_list, b_list, x, alpha_list)
    a_mix_v, b_mix_v = SRK_mixture_parameters_vapour(a_list, b_list, y, alpha_list)

    # solve for Z 
    A_l = a_mix_l * P / (R**2 * T**2)
    B_l = b_mix_l * P / (R * T)
    SRK_eq_liq = (Z_L**3) - (Z_L**2) + ((A_l - B_l - (B_l**2)) * Z_L) - (A_l * B_l)
    eq_liq = sp.N(SRK_eq_liq)
    z_l_roots = sp.solve(eq_liq, Z_L)
    z_l = min(
        z.as_real_imag()[0] for z in z_l_roots
        if abs (z.as_real_imag()[1]) < 1e-6
    )

    A_v = a_mix_v * P / (R**2 * T**2)
    B_v = b_mix_v * P / (R * T)
    SRK_eq_vap = (Z_V**3) - (Z_V**2) + ((A_v - B_v - (B_v**2)) * Z_V) - (A_v * B_v)
    eq_vap = sp.N(SRK_eq_vap)
    z_v_roots = sp.solve(eq_vap, Z_V)
    z_v = max(
        z.as_real_imag()[0] for z in z_v_roots
        if abs (z.as_real_imag()[1]) < 1e-6
    )
    print("compressibility factor of the liquid")
    print(z_l)
    print("compressibility factor of the vapour")
    print(z_v)


    # AA_i values
    AA_i_L = calculate_AA_i(a_list, alpha_list, a_mix_l, x)
    AA_i_V = calculate_AA_i_vapour(a_list, alpha_list, y, a_mix_v)

    # fugacity coefficients
    phi_l = fugacity_srk(b_list, a_list, AA_i_L, z_l, A_l, B_l, b_mix_l, "liquid")
    phi_v = fugacity_srk(b_list, a_list, AA_i_V, z_v, A_v, B_v, b_mix_v, "vapour")

    #update K values 
    new_K_values = [phi_l[i] / phi_v[i] for i in range(len(phi_l))]
    max_diff = max(abs((new_K_values[i] - K_old[i])/K_old[i]) for i in range(len(K_val)))

    print(f"Iter {iteration+1} | Max K diff: {max_diff:.6f}")
    print(f"  Liquid fraction: {L_solution:.6f}")
    
    K_val = new_K_values.copy()
    L_initial_estimate = L_solution
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

#fractional vapour flowrate
V = 1 - L_solution 
print(V)
# molar balance for the VLE
V_flowrate = V * Z_flowrate
L_flowrate = L_solution * Z_flowrate
liquid_component_flowrates = [x[i] * L_flowrate for i in range (len(x))]
Vapour_component_flowrates = [y[i] * V_flowrate for i in range (len(y))]
Z_component_flowrates = [mol_fraction[i] * Z_flowrate for mol_fraction[i] in mol_fraction]

dMB = pd.DataFrame({
    "component name": component_names,
    "feed molar flowrates": Z_component_flowrates,
    "liquid molar flowrates": liquid_component_flowrates,
    "vapour molar flowrates": Vapour_component_flowrates
})
print(dMB)
print("feed flowrate (kmol/hr)")
print(Z_flowrate)
print("vapour flowrate (kmol/hr)")
print(V_flowrate)
print("liquid flowrate (kmol/hr)")
print(L_flowrate)
# mass balance for the VLE based on the molar balance 
Z_mass_flowrate = [Mr[i] * Z_component_flowrates[i] for i in range(len(Mr))]
L_mass_flowrate = [Mr[i] * liquid_component_flowrates[i] for i in range(len(Mr))]
V_mass_flowrate = [Mr[i] * Vapour_component_flowrates[i] for i in range(len(Mr))]
l_mass_flowrate = sum(L_mass_flowrate)
v_mass_flowrate = sum(V_mass_flowrate)
z_mass_flowrate = sum(Z_mass_flowrate)
dMass = pd.DataFrame({
    "component name": component_names,
    "feed mass flowrates":Z_mass_flowrate ,
    "liquid mass flowrates": L_mass_flowrate ,
    "vapour mass flowrates": V_mass_flowrate
})
print(dMass)
print("feed flowrate (kg/hr)")
print(z_mass_flowrate)
print("vapour flowrate (kg/hr)")
print(v_mass_flowrate)
print("liquid flowrate (kg/hr)")
print(l_mass_flowrate)
total_mass_flowrate = l_mass_flowrate + v_mass_flowrate
print(total_mass_flowrate)
# molar mass for the liquid and vapour phases and subsequent densities
M_liquid = sum([Mr[i] * x[i] for i in range(len(Mr))])
M_vapour = sum([Mr[i] * y[i] for i in range(len(Mr))])
l_density = ((P * M_liquid) / (R * T * z_l)) * 1000 
v_density = ((P* M_vapour) / (R * T * z_v)) * 1000

print("molecular mass of the liquid phase")
print(M_liquid)
print("molecular mass of the vapour phase")
print(M_vapour)
print("density of the liquid phase (kg/m^3)")
print(l_density)
print("density of the vapour phase (kg/m^3)")
print(v_density)
