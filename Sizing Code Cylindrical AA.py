import numpy as np
import itertools as it
import math as m

#%% Variables

t1 = np.arange(0.01, 10, 0.1) # Thickness in mm
R1 = np.arange(0.43, 1.2, 0.1) # Radius in m
L1 = np.arange(1.81, 3, 0.1) # Length in m
Mat1 = [["AA 2024,", 2780, 73.1*10**9, 31.5*10**6, 324*10**6], ["AA 7075,", 2810, 71.7*10**9, 17.6*10**6, 503*10**6], ["AA 2014,", 2800, 73.1*10**9, 19*10**6, 414*10**6], ["AA 6061,", 2700, 68.9*10**9, 29*10**6, 276*10**6]]
# Mat: Name, rho, E, K1c, Sig_yield 
combinations = list(it.product(t1, R1, L1, Mat1))
print("\r")
print("Number of combinations considered:", len(combinations), "elements.")
print("\r")

# Making seperate lists o.o. the combination list
t2 = []
R2 = []
L2 = []
Mat2 = []
                      
for i in combinations:  
    k = list(i)
    t2.append(k[0])
    R2.append(k[1])
    L2.append(k[2])
    Mat2.append(k[3])

#%% Constants and Formulas

m_sc = 382.76 # kg                        
g = 9.81 # m/s^2

f_lat_req = 10  # Hz minimum
f_ax_req = 15  # Hz minimum

P_latmax = 3*m_sc*g
N_axmax = 8.5*m_sc*g                  
T = 7603000 # N (Falcon 9 1st stage thrust)

#%% Elimination of combinations

valid_combinations = []
for i in range(len(combinations)):
    # Requirements
    t = t2[i]/1000 # mm to m
    R = R2[i]
    L = L2[i]
    rho = float(Mat2[i][1])
    E = float(Mat2[i][2])
    Sig_y = float(Mat2[i][4])
    
    # Beam properties
    I = m.pi/4*(R**4-(R-t)**4)                          # Area moment of I cros sec
    A = m.pi*(R**2-(R-t)**2)                            # Area of cros sec

    # Launch loads
    Sig_ax = N_axmax / A
    Sig_ax2 = T / A
    Sig_bend = P_latmax*L*R / I
    Tau_lat = 2*P_latmax / A
    
    # Maximum allowed stresses
    Sig_buck = m.pi**2*E/4 * (I/(L*A))**2
    Sig_crax = E*( 9*(t/R)**1.6 + 0.16*(t/L)**1.3)
    
    # Natural frequencies
    f_ax = 0.25 * m.sqrt( (A*E) / (m_sc * L))
    f_lat = 0.56 * m.sqrt( (E*I) / (m_sc * L**3) )
    
    # Mass-volume
    V = A * L
    
    if (Sig_ax <= Sig_y) and (Sig_bend <= Sig_y) and (f_ax > f_ax_req) and (f_lat > f_lat_req) and (Tau_lat <= Sig_y) and (Sig_ax < Sig_crax) and (Sig_ax < Sig_buck) and (Sig_ax2 < Sig_buck) and (Sig_ax2 < Sig_crax):
        mass = rho * V # kg
        mass_rounded = np.around(mass, decimals=3)
        valid_combinations.append([i, mass_rounded]) # index of combinations list, with corresponding mass

# Finding info of index with lowest mass
lowest_mass_index = []
lowest_mass = min(x[1] for x in valid_combinations)
print("Lowest estimated mass is", lowest_mass, "kg.")
for i in valid_combinations:
    if i[1] == lowest_mass:
        lowest_mass_index.append(i[0])

print("This mass appears in the following combinations:")
for i in lowest_mass_index:
    print("Material =", combinations[i][3][0], "Thickness =", np.around(combinations[i][0], decimals=3), "mm, Radius =", np.around(combinations[i][1], decimals=3), "m, Length =", np.around(combinations[i][2], decimals=3), "m")




