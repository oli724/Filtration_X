import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from MCA_parser import MCA


mca = MCA(r"semaine_1\Tension_50kV_5uA.mca")
counts = np.array(mca.DATA)  # Nombre de comptes par canal
energies =  np.load("energie_semaine_1.npy")
live_time = mca.get_live_time() 
dead_time = mca.get_dead_time()  
print("deadtime: ", dead_time)
######## COUNT RATE #################
count_rate = np.sum(counts)/live_time
print(count_rate)


########## ENERGIE MAX #############
THRESHOLD = 150  # Adjust based on noise level
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
    return sum(E*I)/sum(I)
print("Energie moyenne:", mean_energy(energies, counts))
###### PLOT ########     
plt.bar(energies, counts, width=np.median(np.diff(energies)), color="blue", alpha=0.5, label="Spectre")
plt.axhline(THRESHOLD)
plt.show()

