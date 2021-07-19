import numpy as np
k = 1.380649*10**(-23)  # Boltzmann-Konstante

def get_stochastical_rate_constant(d1, d2, T, m1, m2):
    d12 = (d1 + d2) / 2
    m12 = (m1 * m2) / (m1 + m2)
    v12 = np.sqrt( (8*k*T) / (np.pi*m12) )
    c = np.pi*d12**2*v12
    return c

def get_stochastical_rate_constants(reactions, L):
    c = []
    for reaction in reactions:
        c.append(get_stochastical_rate_constant(diameter1=1, diameter2=1, temperature=1, mass1=2, mass2=2))
    return c