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
nist_data_Al = np.loadtxt('NIST_AL_coeff_total.csv', skiprows=2)
nist_data_Mo = np.loadtxt('NIST_Mo_coeff_total.csv', skiprows=2)
nist_data_Cu = np.loadtxt('NIST_Mo_coeff_total.csv', skiprows=2)
nist_data_W = np.loadtxt('NIST_W_coeff_total.csv', skiprows=2)
nist_data_Ag = np.loadtxt('NIST_Ag_coeff_total.csv', skiprows=2)

nist_energy_MeV = nist_data_Al[:, 0]
nist_mu_Al = nist_data_Al[:, 1]

nist_energy_MeV_Mo = nist_data_Mo[:, 0]
nist_mu_Mo = nist_data_Mo[:, 1]   
 
nist_energy_MeV_Cu = nist_data_Cu[:, 0]
nist_mu_Cu = nist_data_Cu[:, 1]    # 2nd column: μ/ρ in cm²/g

nist_energy_MeV_W = nist_data_W[:, 0]
nist_mu_W = nist_data_W[:, 1]

nist_energy_MeV_Ag = nist_data_Ag[:, 0]
nist_mu_Ag= nist_data_Ag[:, 1]
# Convert MeV to keV and filter to 0.5-20 keV range
nist_energy_keV_Al= nist_energy_MeV * 1000  
nist_energy_keV_Mo = nist_energy_MeV_Mo * 1000  
nist_energy_keV_Cu = nist_energy_MeV_Cu * 1000  
nist_energy_keV_W = nist_energy_MeV_W * 1000  
nist_energy_keV_Ag= nist_energy_MeV_Ag * 1000  

energy_min = 14.8
energy_max =15.1
valid_mask_Al = (nist_energy_keV_Al >= energy_min) & (nist_energy_keV_Al <= energy_max)
nist_energy_keV_Al = nist_energy_keV_Al[valid_mask_Al]
nist_mu_Al = nist_mu_Al[valid_mask_Al]

valid_mask_Cu = (nist_energy_keV_Cu >= energy_min) & (nist_energy_keV_Cu <= energy_max)
nist_energy_keV_Cu = nist_energy_keV_Cu[valid_mask_Cu]
nist_mu_Cu = nist_mu_Cu[valid_mask_Cu]

valid_mask_Mo = (nist_energy_keV_Mo >= energy_min) & (nist_energy_keV_Mo <= energy_max)
nist_energy_keV_Mo = nist_energy_keV_Mo[valid_mask_Mo]
nist_mu_Mo = nist_mu_Mo[valid_mask_Mo]

valid_mask_W = (nist_energy_keV_W >= energy_min) & (nist_energy_keV_W <= energy_max)
nist_energy_keV_W = nist_energy_keV_W[valid_mask_W]
nist_mu_W = nist_mu_W[valid_mask_W]

valid_mask_Ag = (nist_energy_keV_Ag >= energy_min) & (nist_energy_keV_Ag <= energy_max)
nist_energy_keV_Ag = nist_energy_keV_Ag[valid_mask_Ag]
nist_mu_Ag = nist_mu_Ag[valid_mask_Ag]

# Verify array lengths match
assert len(nist_energy_keV_Al) == len(nist_mu_Al), "NIST data dimension mismatch"

# Convert to linear attenuation coefficient (cm⁻¹)
rho_Al = 2.7  # g/cm³ density of aluminum
rho_Cu = 8.96
rho_Mo = 10.28
rho_W = 19.25
rho_Ag = 10.49
N_A = constants.Avogadro
nist_tau_Al= nist_mu_Al * (26.982/N_A)
nist_tau_Cu = nist_mu_Cu*(63.546/N_A)
nist_tau_Mo = nist_mu_Mo*(95.95/N_A)
nist_tau_W = nist_mu_W*(107.868/N_A)
nist_tau_Ag = nist_mu_Ag*(183.84/N_A)



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Assuming these are single values per element (not arrays)
mu_vs_Z = np.array([nist_tau_Al[0], nist_tau_Cu[0], nist_tau_Mo[0], nist_tau_Ag[0], nist_tau_W[0]])  # in cm²/g
print(mu_vs_Z)
Z = np.array([13, 29, 42, 47, 74])

def linear_fit(log_Z, a, b):
    return a + b*log_Z

# Apply scaling and log transform
scaled_mu = mu_vs_Z/(1e-24)  # Scaling factor
log_Z = np.log(Z)
log_mu = np.log(scaled_mu)

# Perform the fit
popt, cov = curve_fit(linear_fit, log_Z, log_mu)

# Generate fit line
fit_log_z = np.linspace(min(log_Z), max(log_Z), 100)
fit_line = linear_fit(fit_log_z, *popt)

# Calculate R-squared
residuals = log_mu - linear_fit(log_Z, *popt)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((log_mu - np.mean(log_mu))**2)
r_squared = 1 - (ss_res/ss_tot)

print(popt[0], popt[1])
# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(log_Z, log_mu, label='Data')
plt.plot(fit_log_z, fit_line, 'r-', 
         label=f'Fit: slope={popt[1]:.2f} \nR²={r_squared:.3f}')
print(np.sqrt(np.diag(cov[0])))
# Convert Z ticks back to original values
z_ticks = np.linspace(min(Z), max(Z), 5)
plt.xticks(np.log(z_ticks), labels=[f'{int(z)}' for z in z_ticks])

plt.xlabel('Atomic Number Z (log scale)')
plt.ylabel('log(τ/1e-24)')
plt.title('Mass Attenuation Coefficient vs Atomic Number')
plt.legend()
plt.grid(True)
plt.show()