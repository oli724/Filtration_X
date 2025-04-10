import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import constants
import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MCA_parser import MCA

project_root = Path(__file__).resolve().parent.parent

sys.stdout.reconfigure(encoding='utf-8')
energies = np.load(project_root / "energie_semaine_1.npy")

# Configuration de base
materials = {
    'Al': {'Z': 13, 'A': 26.982, 'rho': 2.7, 'pattern': 'Al'},
    'Cu': {'Z': 29, 'A': 63.546, 'rho': 8.96, 'pattern': 'Cu'},
    'Mo': {'Z': 42, 'A': 95.95, 'rho': 10.28, 'pattern': 'Mo'},
    'Ag': {'Z': 47, 'A': 107.868, 'rho': 10.49, 'pattern': 'Ag'},
    'W': {'Z': 74, 'A': 183.84, 'rho': 19.25, 'pattern': 'W'}
}

def calculate_tau(N, N0, t_cm, rho, A):
    """Calcule le coefficient d'atténuation et son incertitude"""
    N_A = constants.N_A
    tau = (-np.log(N/N0)/(rho*t_cm)) * (A/N_A)
    
    # Calcul de l'incertitude sur tau
    sigma_N = np.sqrt(N)  # Incertitude Poisson sur N
    sigma_N0 = np.sqrt(N0)  # Incertitude Poisson sur N0
    sigma_ratio = np.sqrt((sigma_N/N)**2 + (sigma_N0/N0)**2)  # Incertitude sur N/N0
    sigma_tau = (A/(N_A * rho * t_cm)) * sigma_ratio / (N/N0)  # Propagation d'erreur
    
    return tau, sigma_tau

# Fichier N0
mca_N0 = MCA(project_root / "semaine_1" / "Courant_20kV_25uA.mca")
N0_counts = np.array(mca_N0.DATA)
N0_live_time = mca_N0.get_live_time()
N0_count_rate = N0_counts / N0_live_time

# Nom du matériau à traiter
material = "W"
info = materials[material]

# Épaisseurs en mils
#epaisseurs = np.array([1, 2, 3, 4, 5])
#epaisseurs = np.array([10,20,30,40,50,60,70])
epaisseurs = np.array([1])
tau_values = []
tau_uncertainties = []

for epaisseur in epaisseurs:
    file_path = project_root / "semaine_1" / f"{material}_{epaisseur}mils_20kV_25uA.mca"
    mca = MCA(file_path)
    counts = np.array(mca.DATA)
    live_time = mca.get_live_time()
    count_rate = counts / live_time
    t_cm = epaisseur * 0.00254  # Conversion mils -> cm
    
    # Filtrage en énergie (15 keV)
    energy_mask = (energies >= 14.95) & (energies <= 15.05)
    filtered_N0 = N0_count_rate[energy_mask]
    filtered_counts = count_rate[energy_mask]
    
    # Somme des counts dans la fenêtre d'énergie
    N = np.sum(filtered_counts)
    N0 = np.sum(filtered_N0)
    
    # Calcul de tau et son incertitude
    tau_i, sigma_tau_i = calculate_tau(N, N0, t_cm, info['rho'], info['A'])
    tau_values.append(tau_i)
    tau_uncertainties.append(sigma_tau_i)

# Conversion en array numpy
tau_values = np.array(tau_values)
tau_uncertainties = np.array(tau_uncertainties)

# Moyenne pondérée par les incertitudes
weights = 1 / (tau_uncertainties**2)
tau_mean = np.sum(tau_values * weights) / np.sum(weights)
tau_mean_uncertainty = np.sqrt(1 / np.sum(weights))/300

# Écart-type empirique (pour comparaison)
tau_std = np.std(tau_values)

# Sauvegarde complète
save_data = {
    'material': material,
    'Z': info['Z'],
    'A': info['A'],
    'rho': info['rho'],
    'epaisseurs_mils': epaisseurs,
    'tau': tau_values,
    'tau_uncertainties': tau_uncertainties,
    'tau_mean': tau_mean,
    'tau_mean_uncertainty': tau_mean_uncertainty,
    'tau_std': tau_std,
}

# Affichage des résultats
print(f"\nRésultats pour {material}:")
for i, (epaisseur, tau, sigma) in enumerate(zip(epaisseurs, tau_values, tau_uncertainties)):
    print(f"Épaisseur {epaisseur} mils: τ = {tau:.3e} ± {sigma:.3e} cm²/atom")

print(f"\nMoyenne pondérée: τ = {tau_mean:.3e} ± {tau_mean_uncertainty:.3e} cm²/atom")
print(f"Écart-type empirique: {tau_std:.3e} cm²/atom")

# Sauvegarde des données (décommenter pour utiliser)
np.savez(f"{material}_tau_data_2.npz", **save_data)
