import numpy as np

def m_to_ft(m): 
    ft = m / 0.3048
    return ft

def ft_to_m(ft): 
    m = ft * 0.3048
    return m

def Nm_to_km(Nm):
    
    km = Nm*1.852
    return km
def km_to_Nm(km):
    
    Nm = km/1.852
    return Nm

def deg_to_rad(deg):
    
    rad = deg * np.pi / 180
    return rad

def rad_to_deg(rad):
    
    deg = rad / (np.pi / 180)
    return deg

def degK_to_degC(K):
    
    C = K - 273.15
    return C

def degC_to_degK(C):
    
    K = C + 273.15
    return K

