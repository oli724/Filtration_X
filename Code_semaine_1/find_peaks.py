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

mca = MCA(r"semaine_1\Tension_50kV_5uA.mca")

channels = np.arange(len(mca.DATA))  # Canaux du spectre
counts = np.array(mca.DATA)  # Nombre de comptes par canal

# === 2. Conversion des canaux en énergie ===
energies =  np.load("energie_semaine_1.npy")

# === 3. Détection des pics ===
peaks, _ = find_peaks(counts, height=6000, distance=20)  # Ajuste les seuils selon tes données

# === 4. Définition de la fonction gaussienne ===
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# === 5. Ajustement gaussien pour chaque pic détecté ===
fit_params = []  # Stocker les paramètres des fits
plt.figure(figsize=(10, 5))
plt.bar(energies, counts, width=np.median(np.diff(energies)), color="blue", alpha=0.5, label="Spectre")
print(peaks)
delta_sigma = []
for peak in peaks:
    # Sélection d'une région autour du pic
    window = 10  # Largeur de la région pour le fit
    min_idx = max(0, peak - window)
    max_idx = min(len(counts), peak + window)

    x_fit = energies[min_idx:max_idx]
    y_fit = counts[min_idx:max_idx]

    # Estimation initiale des paramètres (A, mu, sigma)
    A_guess = max(y_fit)
    mu_guess = energies[peak]
    sigma_guess = (energies[max_idx] - energies[min_idx]) / 6  # Approximation empirique

    try:
        popt, cov = curve_fit(gaussian, x_fit, y_fit, p0=[A_guess, mu_guess, sigma_guess])
        fit_params.append(popt)
        delta_sigma.append(np.sqrt(np.diag(cov)[2])
)
        # Tracé des fits gaussiens
        x_smooth = np.linspace(x_fit.min(), x_fit.max(), 100)
        #plt.plot(x_smooth, gaussian(x_smooth, *popt), 'r-', label=f"Fit {mu_guess:.1f} keV")
    
    except RuntimeError:
        print(f" Fit impossible pour le pic à {mu_guess:.1f} keV")

plt.xlabel("Énergie (keV)", fontsize=30)
plt.ylabel("Nombre de Comptes",fontsize=30)
#plt.title("Spectre MCA avec Ajustements Gaussiens")
plt.tick_params(axis='both', which='major', labelsize=20)  # Taille des nombres sur les axes
plt.tick_params(axis='both', which='minor', labelsize=20)  # Taille des nombres des ticks mineurs

plt.legend()
plt.show()

# === 6. Affichage des résultats des fits ===
sigma_list = []
E_list = []

for i, (A, mu, sigma) in enumerate(fit_params):
    print(f" Pic {i+1}: E = {mu:.2f} keV, Largeur (sigma) = {sigma:.2f} keV, FWHM = {2*np.sqrt(2*np.log(2))*sigma:.2f} keV, Amplitude = {A:.1f}")
    sigma_list.append(sigma)
    E_list.append(mu)