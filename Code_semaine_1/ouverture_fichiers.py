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

mca_N0 = MCA(r"semaine_1\Courant_20kV_25uA.mca")
N0_counts = np.array(mca_N0.DATA)
N0_live_time = mca_N0.get_live_time()
N0_count_rate = N0_counts/N0_live_time

mca = MCA(r"semaine_1\Cu_2mils_20kV_25uA.mca")
counts = np.array(mca.DATA)  # Nombre de comptes par canal
energies =  np.load("energie_semaine_1.npy")
live_time = mca.get_live_time() 
dead_time = mca.get_dead_time()  
print("deadtime: ", dead_time)
######## COUNT RATE #################
total_count_rate = np.sum(counts)/live_time
count_rate = counts/live_time
print(total_count_rate)

energy_mask = (energies >= 2) & (energies <= 20)
counts = counts[energy_mask]
N0_counts = N0_counts[energy_mask]
N0_count_rate = N0_count_rate[energy_mask]
count_rate = count_rate[energy_mask]
energies = energies[energy_mask]
########## ENERGIE MAX #############
THRESHOLD = 0  # Adjust based on noise level
# Find the last index where counts are above the threshold
above_threshold = np.where(counts > THRESHOLD)[0]
if len(above_threshold) > 0:
    last_above_threshold = above_threshold[-1]  # Last valid index
    energie_max = energies[last_above_threshold]
    print(f"Maximum energy (last point above threshold): {energie_max}")
else:
    print("No counts above threshold found in the spectrum.")   

########### ENERGIE MOYENNE ###############
def mean_energy(E, I):
    E = np.array(E)
    I = np.array(I)>THRESHOLD
    E_moy = sum(E*I)/sum(I)
    sigma = np.sqrt((np.sum(I*(E-E_moy)**2)))/(np.sum(I))
    return E_moy, sigma
print("Energie moyenne:", mean_energy(energies, counts))
###### PLOT ########     
plt.bar(energies, count_rate, width=np.median(np.diff(energies)), color="blue", alpha=0.5, label="Spectre avec filtre d'Aluminium (10 mils)")

NsurN0 = np.mean(count_rate)/np.mean(N0_count_rate)
print(NsurN0)
#plt.bar(energies, N0_count_rate , width=np.median(np.diff(energies)), color="red", alpha=0.5, label="Spectre sans filtre")
#plt.axhline(THRESHOLD)
#plt.axvline(50)
#plt.axvline(11.28)
plt.ylabel("Nombre de compte \n (échelle logarithmique)",fontsize = 30)
plt.xlabel("Énergie [keV]", fontsize=30)
plt.grid(True, which='both', alpha=0.3)
#plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=20)  # Taille des nombres sur les axes
plt.tick_params(axis='both', which='major', labelsize=20) 
#plt.tight_layout()
plt.legend(fontsize=20,loc='upper right')
plt.show()

