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

#mca_N0 = MCA(r"semaine_1\Courant_20kV_25uA.mca")
mca_N0 = MCA(r"semaine_2\N_0_20kV_25uA_2e_align.mca")

N0_counts = np.array(mca_N0.DATA)
N0_live_time = mca_N0.get_live_time()
N0_count_rate = N0_counts/N0_live_time

mca = MCA(r"semaine_2\Cu-W_20kV_25uA_2e_align.mca")
#mca = MCA(r"semaine_3\Cu_1mils_25kV_25uA_SiPin.mca")
counts = np.array(mca.DATA)  # Nombre de comptes par canal
energies =  np.load("energie_semaine_2.npy")
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
THRESHOLD = 20  # Adjust based on noise level
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

NsurN0 = np.sum(count_rate)/np.sum(N0_count_rate)
sigma_NN_0 = NsurN0*np.sqrt(1/np.sum(count_rate) + 1/np.sum(N0_count_rate))
energy_mean_N0 =mean_energy(energies, N0_counts)
energy_mean =  mean_energy(energies, counts)
print("Energie moyenne avec filtre:", energy_mean)
print("Energie moyenne sans filtre:", energy_mean_N0)
print("rapport moyen NsurN0:", NsurN0, sigma_NN_0)

###### PLOT ########     
plt.bar(energies, count_rate, width=np.median(np.diff(energies)), color="blue", alpha=0.5, label="Spectre avec filtre")



plt.bar(energies, N0_count_rate , width=np.median(np.diff(energies)), color="red", alpha=0.5, label="Spectre sans filtre")

plt.ylabel("Nombre de compte \n (échelle logarithmique)",fontsize = 30)
plt.xlabel("Énergie [keV]", fontsize=30)
plt.grid(True, which='both', alpha=0.3)
plt.yscale("log")
plt.tick_params(axis='both', which='major', labelsize=20)  # Taille des nombres sur les axes
plt.tick_params(axis='both', which='major', labelsize=20) 
plt.axvline(energy_mean[0], color='b')
plt.axvline(energy_mean_N0[0], color = 'r')
plt.fill_betweenx(
    [0, 10000],  # y-range for the uncertainty region (0 to 10000)
    energy_mean[0] - energy_mean[1],  # left bound of the uncertainty region
    energy_mean[0] + energy_mean[1],  # right bound of the uncertainty region
    color='b',  # color for the uncertainty region
    alpha=0.3,  # transparency of the shaded region
    label = f"Energie moyenne avec filtre: {energy_mean[0]:.2f} $\pm$ {energy_mean[1]:.2f} "
)

# Plot the horizontal uncertainty (shaded region) around energy_mean without filter
plt.fill_betweenx(
    [0, 10000],  # y-range for the uncertainty region (0 to 10000)
    energy_mean_N0[0] - energy_mean_N0[1],  # left bound of the uncertainty region
    energy_mean_N0[0] + energy_mean_N0[1],  # right bound of the uncertainty region
    color='r',  # color for the uncertainty region
    alpha=0.3,  # transparency of the shaded region
    label = f"Energie moyenne sans filtre: {energy_mean_N0[0]:.2f} $\pm$ {energy_mean_N0[1]:.2f} "
)
custom_text = f"\n N/N$_0$ = {NsurN0:.4f}"
plt.text(0.95, 0.95, custom_text, transform=plt.gca().transAxes, fontsize=35, verticalalignment='top', horizontalalignment='right', color='black')
plt.legend(fontsize=18,loc='upper left')
plt.show()

