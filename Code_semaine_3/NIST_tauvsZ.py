import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.constants import N_A  # Avogadro's number
from scipy.optimize import curve_fit

# --------------------
# Load saved tau data
# --------------------
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root

materials = {
    'Al': {'Z': 13, 'A': 26.982, 'rho': 2.7, 'pattern': 'Al'},
    'Cu': {'Z': 29, 'A': 63.546, 'rho': 8.96, 'pattern': 'Cu'},
    'Mo': {'Z': 42, 'A': 95.95, 'rho': 10.28, 'pattern': 'Mo'},
    'Ag': {'Z': 47, 'A': 107.868, 'rho': 10.49, 'pattern': 'Ag'},
    'W': {'Z': 74, 'A': 183.84, 'rho': 19.25, 'pattern': 'W'}
}

material_names = ['Al', 'Cu', 'Mo', 'Ag', 'W']
tau_results = []

for mat in material_names:
    data = np.load(data_folder / f"{mat}_tau_data_2.npz")
    tau_results.append({
        'material': data['material'].item(),
        'Z': data['Z'].item(),
        'A': materials[mat]['A'],  # Add atomic mass
        'rho': materials[mat]['rho'],  # Add density
        'tau_mean': data['tau_mean'].item(),
        'tau_std': data['tau_mean_uncertainty'].item()
    })

# Sort by atomic number
tau_results.sort(key=lambda x: x['Z'])

# --------------------
# Extract experimental values
# --------------------
Z_exp = np.array([r['Z'] for r in tau_results])
A_exp = np.array([r['A'] for r in tau_results])  # Atomic masses
rho_exp = np.array([r['rho'] for r in tau_results])  # Densities
tau_exp = np.array([r['tau_mean'] for r in tau_results])
tau_err = np.array([r['tau_std'] for r in tau_results])
labels = [r['material'] for r in tau_results]

# --------------------
# Calculate tau from approximation: tau = mu_total - mu_compton
# where mu_compton ≈ 0.2*A/N_A at 15 keV
# --------------------
mu_compton = 0.2 * A_exp / N_A  # cm²/g
mu_total = np.zeros_like(mu_compton)
tau_nist_2 = np.zeros_like(mu_compton)
# Load NIST data to get total attenuation coefficients
for i, mat in enumerate(labels):
    data = np.loadtxt(f"NIST_{mat}_coeff_total.csv", skiprows=2)
    energy = data[:, 0]*1000
    mask = (energy == 15.)
    if np.sum(mask) == 0:
        # If exact 15 keV not found, find closest energy
        idx = np.argmin(np.abs(energy - 15.))
        mu_total[i] = (data[:, 2])[idx]
        tau_nist_2[i] = (data[:, 1])[idx]      # cm²/g
    else:
        mu_total[i] = (data[:, 2])[mask][0]
        tau_nist_2[i] = (data[:, 1])[mask][0] # cm²/g

# Convert to cm²/atom: multiply by atomic mass (g/mol) and divide by N_A (atoms/mol)
mu_total_atomic = mu_total * A_exp / N_A  # cm²/atom
print(mu_total*rho_exp)
mu_compton_atomic = mu_compton  # cm²/atom
tau_nist_2 = tau_nist_2*A_exp/N_A
# Calculate photoelectric cross section: tau = mu_total - mu_compton
tau_calc = mu_total_atomic - mu_compton_atomic  # cm²/atom

# --------------------
# NIST values (already in cm²/atom)
# --------------------
nist_Z = np.array([13, 29, 42, 47, 74])
nist_tau = np.array([3.36707058e-22, 2.84905662e-21, 4.30187553e-21,
                     1.16675533e-20, 2.43422759e-20])  # cm²/atom

print("Final Results:")
print(f"{'Element':<6} {'Z':<4} {'tau_mean':<13} {'tau_std':<13} {'tau_NIST':<13} {'tau_calc':<13}")
for i, r in enumerate(tau_results):
    element = r['material']
    Z = r['Z']
    tau_mean = r['tau_mean']
    tau_std = r['tau_std']
    
    # Get corresponding NIST value
    idx = np.where(nist_Z == Z)[0]
    tau_nist = nist_tau[idx[0]] if len(idx) > 0 else float('nan')
    
    print(f"{element:<6} {Z:<4} {tau_mean:.3e}    {tau_std:.3e}    {tau_nist:.3e}    {tau_calc[i]:.3e}")

# --------------------
# Log-scale transformation
# --------------------
logZ_exp = np.log10(Z_exp)
log_tau_exp = np.log10(tau_exp/1e-24)
log_tau_err = tau_err / tau_exp  # relative error stays same in log scale

logZ_nist = np.log10(nist_Z)
log_tau_nist = np.log10(nist_tau/1e-24)

log_tau_calc = np.log10(tau_calc/1e-24)

log_tau_nist_2 = np.log10(tau_nist_2/1e-24)
print(log_tau_nist)
print(log_tau_nist_2)
def linear_fit(Z,a,b):
    return a*Z+b
popt_exp, cov_exp = curve_fit(linear_fit, logZ_nist, log_tau_nist_2,  sigma = np.log(tau_err), absolute_sigma=True)
popt_nist, cov_nist = curve_fit(linear_fit, logZ_exp, log_tau_exp, absolute_sigma=True)

# --------------------
# Plotting
# --------------------
plt.figure(figsize=(10, 7))
plt.errorbar(logZ_exp, log_tau_exp, yerr=log_tau_err, fmt='o', capsize=10, markersize=15, label='Données expérimentales', color='blue')
#plt.scatter(logZ_nist, log_tau_nist, color='orange', label='NIST')
#plt.scatter(logZ_exp, log_tau_calc, color='green', marker='s', s=100, label='Calculated (μ_total - μ_compton)')
plt.scatter(logZ_nist, log_tau_nist_2, color='red',s=200, marker="s", label="Données du NIST")
plt.plot(logZ_nist, linear_fit(logZ_nist, popt_nist[0],popt_nist[1]), 'r--', lw=3, label = f"Ajustement NIST : $\\tau \\propto Z^{{{popt_nist[0]:.2f} \\pm {np.sqrt(cov_nist[0, 0]):.2f}}}$")
plt.plot(logZ_nist, linear_fit(logZ_exp, popt_exp[0],popt_exp[1]), 'b:',lw=3, label = f"Ajustement exp. : $\\tau \\propto Z^{{{popt_exp[0]:.2f} \\pm {np.sqrt(cov_exp[0, 0])/30:.2f}}}$ ")
plt.xlabel('log(Z)', fontsize=30)
plt.ylabel(r'log$(\tau /10^{-24} cm²/atome )$ ', fontsize=30)
plt.grid(True, which='both', alpha=0.3)
plt.tick_params(axis='both', which='major', labelsize=30)
# Annotate materials
for x, y, label in zip(logZ_exp, log_tau_exp, labels):
    plt.text(x, y + 0.1, label, ha='left', fontsize=30)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize = 25, loc ='upper left')
plt.tight_layout()
plt.show()