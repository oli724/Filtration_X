import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress  # For R² calculation

# Your data
courant = [5, 10, 15, 20, 25]
tension = [10, 20, 30, 40, 50]
freq_courant = [12413, 24059, 34923, 45371, 55483]
freq_tension = [394, 12408, 41592, 82490, 118057]
error = [9,16,21,39,42]

# Linear fit function
def lineaire(x, a):
    return a * x 

# Perform the fit
popt, cov = curve_fit(lineaire, courant, freq_courant)
a_fit = popt[0]

# Calculate R² using scipy's linregress
slope, intercept, r_value, p_value, std_err = linregress(courant, freq_courant)
r_squared = r_value**2
print(slope)
print(intercept)
# Generate fit line
fit_x = np.linspace(0, 25, 100)
lin_fit = lineaire(fit_x, a_fit)

# Create plot
plt.figure(figsize=(12, 8))

# Plot data and fit
plt.errorbar(courant, freq_courant,yerr = error,fmt='o', markersize=8,
                capsize=5, label="Données expérimentales")
plt.plot(fit_x, lin_fit, 'r-', 
         label=f"Ajustement linéaire: ({a_fit:.0f} ± {std_err:.0f}) compte/(s·μA) × courant\n"
      f'R² = {r_squared:.4f}',
         linewidth=2)

# Formatting
plt.xlabel("Courant [μA]", fontsize=30)
plt.ylabel("Fréquence de comptage [compte/s]", fontsize=30)
plt.legend(fontsize=20, loc='upper left')
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Adjust tick sizes
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)

plt.tight_layout()
plt.show()