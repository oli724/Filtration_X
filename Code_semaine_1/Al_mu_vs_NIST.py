import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import sys
import os
from pathlib import Path

# Ajouter le dossier parent au chemin Python
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
from MCA_parser import MCA
from scipy import constants


#nist_data = np.loadtxt('NIST_AL_coeff_total.csv', skiprows=2)  # Skip 2 header lines
nist_data = np.loadtxt('NIST_Mo_coeff_total.csv', skiprows=2)
nist_energy_MeV = nist_data[:, 0]  # 1st column: Energy in MeV
nist_mu = nist_data[:, 1]          # 2nd column: μ/ρ in cm²/g

# Convert MeV to keV and filter to 0.5-20 keV range
nist_energy_keV = nist_energy_MeV * 1000  
valid_mask = (nist_energy_keV >= 10) & (nist_energy_keV <= 18)
nist_energy_keV = nist_energy_keV[valid_mask]
nist_mu = nist_mu[valid_mask]

# Verify array lengths match
assert len(nist_energy_keV) == len(nist_mu), "NIST data dimension mismatch"

# Convert to linear attenuation coefficient (cm⁻¹)
rho_Al = 10.28  # g/cm³ density of aluminum
nist_tau = nist_mu * rho_Al
print(np.sum(nist_tau)/len(nist_tau))
plt.plot(nist_energy_keV, nist_tau)
plt.show()