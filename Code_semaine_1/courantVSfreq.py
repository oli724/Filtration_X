import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

courant = [5,10,15,20,25]
tension = [10,20,30,40,50]
freq_courant = [12413,24059,34923,45371,55483]
freq_tension = [394,12408,41592,82490,118057]
def lineaire(x,a,b):
    return a*x_b
popt, cov = curve_fit(lineaire, courant,freq_courant)
a_fit, b_fit = popt[0], popt[1]
plt.scatter(courant, freq_courant, label = "courant")
#plt.plot(tension, freq_tension, label = "tension")
plt.axtitle
plt.legend()
plt.show()