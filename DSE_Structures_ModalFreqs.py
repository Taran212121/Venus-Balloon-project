
import numpy as np

k = 5 # n+1 bodies, 1 balloon, ... tether, 1 gandola
Ndof = k+1 # y1+angles

y1 = 0
z1 = 0
lamb = 0.2

q = [y1]
theta = np.array([5, 10, 15, 20, 25])/180*np.pi

for i in range(k):   # 0 to k-1
    q.append(theta[i])
print("Thetax = ", q)

m = np.array([50, 2, 2, 2, 10])
L = np.array([5, 1, 1, 1, 0.6])
rho = np.array([0.4, 0.5, 0.5, 0.5, 0.8])
I = np.array([3, 0.2, 0.2, 0.2, 1])
g = 8.7  # Venus gravity at 55km alt

def mu(k): # input k is according to python, k=0:balloon
    mu=0
    for i in range(k, len(m)):
        mu+=m[i]
    return mu
print(mu(3))

buoy = -mu(1)*g

#%% yk and zk

def y(k): # k=0:balloon, k=1:1st teather
    y = y1
    if k > 0:
        # first term:
        y += (1 - rho[0]) * L[0] * np.sin(theta[0])
    
        # middle sum:
        for i in range(1, k): # python stops at k-1
            y += L[i] * np.sin(theta[i])
    
        # last term:
        y += rho[k] * L[k] * np.sin(theta[k])

    return y

def ydot(k, thetadot):
    # first term
    ydot = (1-rho[0]) * L[0] * np.cos(theta[0]) * thetadot[0]
    
    # middle term
    for i in range(1, k):
        ydot += L[i] * np.cos(theta[i])*thetadot[i]
        
    # last term
    ydot += rho[k] * L[k] * np.cos(theta[k]) * thetadot[k]    
    
    return ydot

def z(k):
    z = z1
    if k > 0:
        # first term:
        z += -(1 - rho[0]) * L[0] * np.cos(theta[0])
        
        # middle sum:
        for i in range(1, k):
            z += -L[i] * np.cos(theta[i])
            
        # last term:
        z += -rho[k] * L[k] * np.cos(theta[k])
        
    return z

def zdot(k):
    return 0

def Ep():
    Ep = -mu(0)*g*(z(0) - lamb* L[0] * np.cos(theta[0]))
    for i in range(0, k+1):
        Ep += m[i] * g * z(i)
    
    return Ep
 
for i in np.arange(0, k):
    print(y(i), z(i))


