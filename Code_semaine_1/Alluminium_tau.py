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

mca_N0 = MCA(r"semaine_1\Courant_20kV_25uA.mca")
N0_counts = np.array(mca_N0.DATA)
N0_live_time = mca_N0.get_live_time() 
N0_count_rate = N0_counts/N0_live_time
energies =  np.load("energie_semaine_1.npy")
#t_mils = np.array([10,20,30,40,50,60,70])
t_mils = np.array([1,2,3,4,5])
mils_to_cm = 0.00254 #cm/mils

##### TAU ######

materials = {
    'Al': {'Z': 13, 'A': 26.982, 'rho': 2.7, 'pattern': 'Al'},
    'Cu': {'Z': 29, 'A': 63.546, 'rho': 8.96, 'pattern': 'Cu'},
    'Mo': {'Z': 42, 'A': 95.95, 'rho': 10.28, 'pattern': 'Mo'},
    'Ag': {'Z': 47, 'A': 107.868, 'rho': 10.49, 'pattern': 'Ag'},
    'W': {'Z': 74, 'A': 183.84, 'rho': 19.25, 'pattern': 'W'}
}
nist_mu = [  21.4785,  663.488,   293.3912,  419.3902, 2673.825 ]
exp_mu = [ 23.1038, 703.57, 283.2843]
exp_err = [ 2.5257, 73.76,21.9027]
material = 'Mo'
A = materials[material]['A']
rho = materials[material]['rho']
def tau(t,N,N_0= N0_counts,rho = rho,A = A):
    N_A = constants.N_A
    
    return (-np.ln(N/N0)/(rho*t))*(A/N_A) - 0.20*(A/N_A)
NsurN0 = []
def tau_Al(t, mu):
    return np.e**(-mu*t)

# Define threshold (e.g., 5% of max counts)
THRESHOLD_PERCENT = 0  # Adjust this value as needed

NsurN0 = []
valid_energies = []  
energy_min = 14.95  # keV
energy_max = 15.05  # keV
NsurN0_std = []
for epaisseur in t_mils:
    mca = MCA(f"semaine_1\{material}_{epaisseur}mils_20kV_25uA.mca")
    counts = np.array(mca.DATA)
    live_time = mca.get_live_time()
    dead_time = mca.get_dead_time()
    print(f"Thickness {epaisseur}mils - deadtime: {dead_time:.2f}%")
    
    count_rate = counts / live_time
    
    # Masque pour les énergies dans [7.5, 18] keV
    energy_mask = (energies >= energy_min) & (energies <= energy_max)
    
    # Appliquer le masque d'énergie ET le seuil de comptage
    combined_mask = energy_mask & (count_rate > (np.max(count_rate) * THRESHOLD_PERCENT / 100))
    
    valid_count_rate = count_rate[combined_mask]
    valid_N0_rate = N0_count_rate[combined_mask]
    valid_energy = energies[combined_mask]  # Pour vérification
    # Calcul du ratio N/N0 (en évitant les divisions par zéro)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where((valid_N0_rate > 0) & (valid_count_rate > 0),
                        valid_count_rate / valid_N0_rate,
                        np.nan)
    
    # Stocker la plage d'énergie utilisée (pour débogage)
    if len(valid_energy) > 0:
        valid_energies.append((valid_energy.min(), valid_energy.max()))
    else:
        valid_energies.append((np.nan, np.nan))
    
    # Moyenne des ratios valides (en ignorant NaN)
    valid_ratio = ratio[np.isfinite(ratio)]
    mean_ratio = np.mean(valid_ratio) if len(valid_ratio) > 0 else np.nan
    NsurN0.append(mean_ratio)
    #NsurN0_std.append(np.std(valid_ratio)/np.sqrt(np.mean(valid_count_rate)))
    NsurN0_std.append(mean_ratio*np.sqrt(1/np.mean(valid_count_rate) + 1/np.mean(valid_N0_rate)))
    
# Convert to numpy arrays
NsurN0 = np.array(NsurN0)
t_cm = np.array(t_mils) * mils_to_cm
def mu(NsurN_0, t):
    return -(1/t)*np.log(NsurN_0)
print(mu(NsurN0, t_cm))
# Perform the fit
params, cov = curve_fit(tau_Al, t_cm, NsurN0, sigma = NsurN0_std, absolute_sigma=True)
mu_fit = params[0]
err = np.sqrt(np.diag(cov))[0]
print(f"Fitted attenuation coefficient: {mu_fit:.4f} cm-1 $\pm$ {err:.4f}")

# Generate fit curve
fit_energie = np.linspace(0, t_mils[-1]*mils_to_cm, 500)
fit_tau_Al = tau_Al(fit_energie, mu_fit)

# Plot results
plt.figure(figsize=(10, 6))
plt.errorbar(t_cm, NsurN0, yerr=NsurN0_std,fmt='o', capsize=5, label='Données expérimentales')
plt.plot(fit_energie, fit_tau_Al, 'r-', 
         label=f'Ajustement (Beer-Lambert): μ = [{mu_fit:.2f} $\pm$ {err:.2f}] cm⁻¹')
plt.xlabel('Épaisseur du filtre t (cm)', fontsize = 30)
plt.ylabel('Ratio N/N₀ moyen',fontsize = 30)
#plt.title(f'X-ray Attenuation through Aluminum (Threshold: {THRESHOLD_PERCENT}% of max)')
plt.legend(fontsize =20)
plt.tick_params(axis='both', which='major', labelsize=20)  # Taille des nombres sur les axes
plt.tick_params(axis='both', which='minor', labelsize=20) 
plt.grid(True, which='both', linestyle='--', alpha=0.3)
plt.legend(fontsize = 20, loc="upper right")
plt.show()

# Print energy ranges used
print("\nEnergy ranges used for each thickness:")
for i, (e_min, e_max) in enumerate(valid_energies):
    print(f"{t_mils[i]} mils: {e_min:.2f}-{e_max:.2f} keV")

print("couche de demi atténuation")