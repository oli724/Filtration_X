import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import sys
import os
from pathlib import Path
from scipy import constants

rho_Al = 2.7  # g/cm³ density of aluminum
rho_Cu = 8.96
rho_Mo = 10.28
rho_W = 19.25
rho_Ag = 10.49
N_A = constants.Avogadro


nist_data_Cu = np.loadtxt('NIST_Mo_coeff_total.csv', skiprows=2)
nist_data_Al = np.loadtxt('NIST_Al_coeff_total.csv', skiprows=2)
nist_data_Mo = np.loadtxt('NIST_Mo_coeff_total.csv', skiprows=2)

nist_energy_keV = nist_data_Cu[:, 0]*1000
nist_mu_Cu = nist_data_Cu[:, 2]*rho_Cu
valid_mask = (nist_energy_keV >= 0.5) & (nist_energy_keV <= 30)
nist_energy_keV = nist_energy_keV[valid_mask]
nist_mu_Cu = nist_mu_Cu[valid_mask]


plt.plot(nist_energy_keV,nist_mu_Cu)
plt.yscale("log")

plt.show()


