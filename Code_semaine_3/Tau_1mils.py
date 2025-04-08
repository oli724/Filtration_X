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
    """Calcule le coefficient d'atténuation"""
    N_A = constants.N_A
    return (-np.log(N/N0)/(rho*t_cm)) * (A/N_A)

# Fichier N0
mca_N0 = MCA(project_root / "semaine_1" / "Courant_20kV_25uA.mca")
#mca_N0 = MCA(project_root / "semaine_3" / "N_0_25kV_25uA_SiPin.mca")
N0_counts = np.array(mca_N0.DATA)
N0_live_time = mca_N0.get_live_time()
N0_count_rate = N0_counts / N0_live_time

# Nom du matériau à traiter
material = "Cu"
info = materials[material]

# Épaisseurs en mils
#epaisseurs = np.array([10, 20, 30, 40, 50, 60, 70])
epaisseurs = np.array([1,2,3,4,5])
#epaisseurs = np.array([1])
tau = []

for epaisseur in epaisseurs:
    file_path = project_root / "semaine_1" / f"{material}_{epaisseur}mils_20kV_25uA.mca"
    #file_path = project_root / "semaine_3" / f"{material}_{epaisseur}mils_25kV_25uA_SiPin.mca"
    mca = MCA(file_path)
    counts = np.array(mca.DATA)
    live_time = mca.get_live_time()
    count_rate = counts / live_time
    t_cm = epaisseur * 0.00254
    
    # Filtrage en énergie (15.9-16 keV)
    energy_mask = (energies >= 14.95) & (energies <= 15.05)
    filtered_N0 = N0_count_rate[energy_mask]
    filtered_counts = count_rate[energy_mask]
    NsurN0 = np.sum(filtered_counts) / np.sum(filtered_N0)

    tau_i = calculate_tau(NsurN0, 1.0, t_cm, info['rho'], info['A'])
    tau.append(tau_i)

tau = np.array(tau)
tau_mean = np.mean(tau)
tau_std = np.std(tau)

# Sauvegarde complète
save_data = {
    'material': material,
    'Z': info['Z'],
    'A': info['A'],
    'rho': info['rho'],
    'epaisseurs_mils': epaisseurs,
    'tau': tau,
    'tau_mean': tau_mean,
    'tau_std': tau_std,
}

#np.savez(f"{material}_tau_data_2.npz", **save_data)

print(f"{material} τ_mean: {tau_mean:.3e}, τ_std: {tau_std:.3e}")
print(tau)
