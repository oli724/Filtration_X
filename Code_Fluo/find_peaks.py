import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import sys
import os
from pathlib import Path
import pandas as pd
import datetime

parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)
from MCA_parser import MCA


# Configuration des polices
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# Chemin du fichier MCA (à adapter)
mca_file = r"Fluo\Fluo_clé_00101_30kv_200uA.mca"
alloy_name = "clé_00101"

# Création du dossier de résultats
results_dir = f"Results_fluo\Results_{alloy_name}"
os.makedirs(results_dir, exist_ok=True)

# Chargement des données
mca = MCA(mca_file)
channels = np.arange(len(mca.DATA))
counts = np.array(mca.DATA)
energies = np.load("energie_fluo.npy")

# Détection des pics (paramètres ajustables)
peaks, properties = find_peaks(counts, 
                             height=0.1*max(counts),
                             prominence=0.1*max(counts),
                             distance=15,
                             width=3)

# Fonction gaussienne pour l'ajustement
def gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Initialisation des résultats
results = []

# Création de la figure
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(energies, counts, width=np.median(np.diff(energies)), color='skyblue', alpha=0.7, label='Spectre')

# Ajustement des pics et annotation
colors = plt.cm.tab10(np.linspace(0, 1, len(peaks)))
for i, (peak, color) in enumerate(zip(peaks, colors)):
    window = 15
    min_idx = max(0, peak - window)
    max_idx = min(len(counts), peak + window)
    
    x_fit = energies[min_idx:max_idx]
    y_fit = counts[min_idx:max_idx]
    
    try:
        popt, pcov = curve_fit(gaussian, x_fit, y_fit, 
                              p0=[counts[peak], energies[peak], 0.1])
        
        # Stockage des résultats
        fwhm = 2*np.sqrt(2*np.log(2))*popt[2]
        results.append({
            'Pic': i+1,
            'Energie (keV)': popt[1],
            'Sigma (keV)': popt[2],
            'FWHM (keV)': fwhm,
            'Amplitude': popt[0],
            'Intensite Relative': popt[0]/max(counts)
        })
        
        # Tracé du fit
        x_plot = np.linspace(x_fit.min(), x_fit.max(), 100)
        ax.plot(x_plot, gaussian(x_plot, *popt), '-', color=color, linewidth=2)
        ax.scatter(popt[1], gaussian(popt[1], *popt), color=color, s=80, zorder=5)
        
        # Annotation
        ax.annotate(f'{popt[1]:.1f} keV', 
                   xy=(popt[1], gaussian(popt[1], *popt)),
                   xytext=(5, 10),
                   textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.8))
    
    except Exception as e:
        print(f"Erreur pic {i+1}: {str(e)}")

# Finalisation du graphique
ax.set_xlabel('Énergie (keV)')
ax.set_ylabel('Nombre de comptes')
ax.set_title(f"Spectre XRF - {alloy_name}")
ax.grid(True, linestyle='--', alpha=0.3)
ax.legend()
ax.set_xlim(left=0, right=1.8*energies[peaks[-1]])

# Sauvegarde de la figure
plot_path = os.path.join(results_dir, f"spectre_{alloy_name}.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

# Création et sauvegarde du tableau de résultats
df = pd.DataFrame(results)
csv_path = os.path.join(results_dir, f"resultats_pics_{alloy_name}.csv")
df.to_csv(csv_path, index=False, encoding='utf-8-sig')

# Affichage des résultats
print(f"\nRésultats sauvegardés dans : {results_dir}")
print("\nTableau des pics détectés :")
print(df.to_string(index=False))