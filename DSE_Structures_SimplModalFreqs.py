import numpy as np

# model for modal freqs for double pendulum,
# assumes point masses, massless links, no rotational intertia

m_b = 200   # kg    balloon mass
m_t = 0.2   # kg    tehter mass
m_g = 50    # kg    gondola mass

L_b = 5     # m     balloon radius
L_t = 0.5   # m     tether length
L_g = 0.6   # m     gandola length

L2 = L_b/2 + L_t/2
L3 = L_t/2 + L_g/2

g = 8.7  # Venus gravity at 55km alt


M = np.array([[m_t, 0],
             [0, m_g]])

K = g * np.array( [[(m_t+m_g)/L2 + m_g/L3,   -m_g/L3],
                  [-m_g/L3,                 m_g/L3]])

# generalized eigenvalue problem
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(M) @ K)

# natural frequencies [rad/s]
omega = np.sqrt(eigvals)

print("omega =", omega)
