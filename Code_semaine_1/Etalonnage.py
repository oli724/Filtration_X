import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from MCA_parser import MCA
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

mca = MCA(r"semaine_2\etalonnage_241Am_2.mca")

list(mca.sections.keys()), mca.DATA[:20] if "DATA" in mca.sections else "No DATA section"
 
data = mca.DATA
canaux = range(len(data))
#### semaine 1 ############
#x= [419, 516, 1756]
#y=[13.95,17.74,59.54]
###### semaine 2 ##########
x= [628,788,2621]
y=[13.95,17.74,59.54]
###########################
def lineaire(x,a,b):
    return a*x+b
params, cov = curve_fit(lineaire,x,y)
a, b = params
sigma_a, sigma_b = np.sqrt(np.diag(cov))
cov_ab = cov[0, 1]
#np.save("etalonnage.npy", np.array(([a,b,sigma_a,sigma_b, cov_ab])))
print(params)
canaux_to_energie = lineaire(canaux, params[0],params[1])
np.save("energie_semaine_2.npy" , np.array(canaux_to_energie))
#plt.figure(figsize=(10, 5))
#plt.bar(canaux_to_energie, mca.DATA, width=params[0], color="blue")
plt.scatter(x,y, label = "énergies de référence \n pour le $^{241}$Am", s=200)
plt.plot(np.linspace(0,500,1000), lineaire(np.linspace(0,500,1000), a,b), linestyle = "--", color = "black", label = f" E = {a:.4f}$ \\times$ (canal) - {np.abs(b):.4f}")
plt.xlabel("Position en terme de canal [-]", fontsize = 30)
plt.ylabel("Énergie [keV]",fontsize = 30)
plt.legend(fontsize= 30)
#plt.show()