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

# === 1. Chargement des données ===
mca = MCA(r"semaine_1\Tension_50kV_5uA.mca")
channels = np.arange(len(mca.DATA))  # Canaux du spectre
counts = np.array(mca.DATA)  # Nombre de comptes par canal

# === 2. Conversion des canaux en énergie ===
energies = np.load("energie_semaine_1.npy")

# === 3. Pics caractéristiques du Tungstène (à annoter) ===
tungsten_peaks = {
    "Lα": 8.39,  # Raie Lα
    "Lβ": 9.67,   # Raie Lβ
    "Lγ": 11.28   # Raie Lγ
}

# === 4. Détection des pics expérimentaux ===
peaks, _ = find_peaks(counts, height=6000, distance=20)  # Ajuste les seuils selon tes données

# === 5. Définition de la fonction gaussienne ===
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# === 6. Ajustement gaussien et tracé ===
plt.figure(figsize=(12, 6))
plt.bar(energies, counts, width=np.median(np.diff(energies)), color="blue", alpha=0.5, label="Spectre expérimental")

# Ajustement des pics détectés (votre code original)
fit_params = []
delta_sigma = []
for peak in peaks:
    window = 10
    min_idx = max(0, peak - window)
    max_idx = min(len(counts), peak + window)
    x_fit = energies[min_idx:max_idx]
    y_fit = counts[min_idx:max_idx]
    A_guess = max(y_fit)
    mu_guess = energies[peak]
    sigma_guess = (energies[max_idx] - energies[min_idx]) / 6
    try:
        popt, cov = curve_fit(gaussian, x_fit, y_fit, p0=[A_guess, mu_guess, sigma_guess])
        fit_params.append(popt)
        delta_sigma.append(np.sqrt(np.diag(cov)[2]))
        x_smooth = np.linspace(x_fit.min(), x_fit.max(), 100)
        #plt.plot(x_smooth, gaussian(x_smooth, *popt), 'r-', linewidth=2, label="Ajustement gaussien" if peak == peaks[0] else "")
    except RuntimeError:
        print(f"Fit impossible pour le pic à {mu_guess:.1f} keV")

# === 7. Annotation des pics caractéristiques du Tungstène ===
# === Annotation des pics caractéristiques du Tungstène ===
# === Annotation des pics caractéristiques du Tungstène ===
for i, (peak_name, peak_energy) in enumerate(tungsten_peaks.items()):
    # Find the closest channel to the peak energy
    idx = np.argmin(np.abs(energies - peak_energy))
    y_pos = counts[idx] * 1.15  # Slightly above the peak
    
    # Adjust horizontal alignment and x-offset to avoid overlap
    if i == 0:  # First peak (Lα)
        plt.text(peak_energy - 0.2, y_pos, f"{peak_name} ({peak_energy} keV)", 
                 ha='right', va='bottom', fontsize=15, color='green')
    else:  # Other peaks (Lβ, Lγ)
        plt.text(peak_energy + 0.2, y_pos, f"{peak_name} ({peak_energy} keV)", 
                 ha='left', va='bottom', fontsize=15, color='green')
    
    # Add a faint vertical line at the peak position
    #plt.axvline(x=peak_energy, color='green', linestyle='--', alpha=0.3, linewidth=1)

# === 8. Mise en forme du graphique ===
#plt.title("Spectre de rayons X avec pics caractéristiques du Tungstène", fontsize=16)

plt.grid(True, alpha=0.3)
plt.axvline(50, label = "Énergie maximale théorique")
plt.xlabel("Énergie (keV)", fontsize=30)
plt.ylabel("Nombre de Comptes",fontsize=30)
#plt.title("Spectre MCA avec Ajustements Gaussiens")
plt.tick_params(axis='both', which='major', labelsize=20)  # Taille des nombres sur les axes
plt.tick_params(axis='both', which='minor', labelsize=20) 
plt.tight_layout()
plt.legend(fontsize=30)
plt.show()

# === 9. Affichage des résultats des fits (votre code original) ===
print("\nRésultats des ajustements gaussiens :")
for i, (A, mu, sigma) in enumerate(fit_params):
    print(f"Pic {i+1}: E = {mu:.2f} keV, FWHM = {2.355*sigma:.2f} keV")