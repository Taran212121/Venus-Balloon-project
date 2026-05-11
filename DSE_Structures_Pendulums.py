import numpy as np


#==================================
# INPUT PARAMETERS
#==================================
m1 = 200    # kg    balloon mass
m2 = 0.2    # kg    tether mass
m3 = 50     # kg    gandola mass

L1 = 5      # m     diameter of balloon (considered as thisn sphere)
L2 = 1      # m     length of tether
L3 = 0.6    # m     length of gandola (considered as thin cylinder)
R3 = 0.2    # m     radius of gandola (considered as thin cylinder)

N_tether = 5      # total elements of tether, at least 1

#%%
# total N, now including balloon and gandola
N = N_tether + 2
g = 8.7  # Venus gravity at 55km alt
lamb = 0.1 # related to distance between CG and CP of balloon

# M,L,I lists
m = [m1]
L = [L1]
I = [1/6*m1*L1**2] # thin walled sphere
for i in np.arange(0,N_tether):
    m.append(m2/N_tether)
    L.append(L2/N_tether)
    I.append(1/12*m2/N_tether*(L2/N_tether)**2) # rod perpendicular axis
m.append(m3)
L.append(L3)
I.append(1/2*m3*R3**2 + 1/12*m3*L3**2) # thin walled cylinder

theta = [0.1, 0.2, 0.3, 0.4]

#%% Functions
            
def mu(k):
    # gives the mu value of kth element
    # for k=0:balloon, gives total system mass, for k=N-1:gandola, gives gandola mass
    # mu(0) = m0+m1+m2+...+m{N-1}, mu(1)=m1+m2+m3+...+m{N-1}
    # if statement to make sure mu{N}=0
    mu=0
    if k < N:
        for i in np.arange(k, N):
            mu += m[i]
        return mu
    else:
        return 0
    
#%%

DOF = N+1 # degrees of freedom, y1, theta1, ...

# modify L1 special case, needs (1-rho) factor in matrix
Llist = L.copy()
rho = 1/2
Llist[0] = (1-rho)*L[0] 

mulist = []
for i in range(N+1):
    mulist.append(mu(i))

#%% Mass matrix
Mx = np.zeros((DOF, DOF))
for i in range(DOF):
    if i == 0: # first row/col except first 2x2
        for j in range(2, DOF):
            Mx[i,j] = (1/2*m[j-1]+mulist[j])*Llist[j-1]
    elif i >= 1: # diagonal entries
        Mx[i, i] = (I[i-1] + ((rho**2) * m[i-1] + mulist[i]) * (Llist[i-1] ** 2))
        for j in range(2, DOF):
            if j > i:
                Mx[i,j] = (1/2*m[j-1]+mulist[j])*Llist[i-1]*Llist[j-1]
        
# hard code deviating first 2 rows/cols
Mx[0,0] = mu(0)
Mx[0,1] = Llist[0]*mu(1)
Mx[1,1] = I[0] + Llist[0]**2*mu(1)

for i in range(DOF):
    for j in range (DOF):
        Mx[j,i] = Mx[i,j]

print("Mass matrix:\n", Mx)

#%% K matrix
Kx = np.zeros((DOF, DOF))
for i in range(DOF):
    if i == 0:
        Kx[i,i]=0
    elif i == 1:
        Kx[i,i] = (1-rho)*g*L[0]*mulist[1] - lamb*g*L[0]*mulist[0]
    else:
        Kx[i,i] = (rho*m[i-1]+mulist[i])*g*L[i-1]

print("Stiffness Matrix:\n", Kx)
        
#%% generalized eigenvalue problem
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Mx) @ Kx)

# natural frequencies 
omega = np.sqrt(eigvals)
f = omega/(2*np.pi) # rad/s to Hz

print("Omega =", f, "Hz")







