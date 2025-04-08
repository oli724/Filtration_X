import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------
# Load saved tau data
# --------------------
project_root = Path(__file__).resolve().parent.parent
data_folder = project_root
materials = ['Al', 'Cu', 'Mo', 'Ag', 'W']
tau_results = []

for mat in materials:
    data = np.load(data_folder / f"{mat}_tau_data.npz")
    tau_results.append({
        'material': data['material'].item(),
        'Z': data['Z'].item(),
        'tau_mean': data['tau_mean'].item(),
        'tau_std': data['tau_std'].item()
    })

# Sort by atomic number
tau_results.sort(key=lambda x: x['Z'])

# --------------------
# Extract experimental values
# --------------------
Z_exp = np.array([r['Z'] for r in tau_results])
tau_exp = np.array([r['tau_mean'] for r in tau_results])
tau_err = np.array([r['tau_std'] for r in tau_results])
labels = [r['material'] for r in tau_results]



# --------------------
# NIST values (already in cm²/atom)
# --------------------
nist_Z = np.array([13, 29, 42, 47, 74])
nist_tau = np.array([3.36707058e-22, 2.84905662e-21, 4.30187553e-21,
                     1.16675533e-20, 2.43422759e-20]) / 1e-24  # normalize by 1e-24


print("Final Results:")
print(f"{'Element':<6} {'Z':<4} {'tau_mean':<13} {'tau_std':<13} {'tau_NIST':<13}")
for r in tau_results:
    element = r['material']
    Z = r['Z']
    tau_mean = r['tau_mean']
    tau_std = r['tau_std']
    
    # Get corresponding NIST value
    idx = np.where(nist_Z == Z)[0]
    tau_nist = (nist_tau*1e-24)[idx[0]] if len(idx) > 0 else float('nan')
    
    print(f"{element:<6} {Z:<4} {tau_mean:.3e}    {tau_std:.3e}    {tau_nist:.3e}")
# --------------------
# Log-scale transformation
# --------------------
logZ_exp = np.log(Z_exp)
log_tau_exp = np.log(tau_exp / 1e-24)
log_tau_err = tau_err / tau_exp  # relative error stays same in log scale

logZ_nist = np.log(nist_Z)
log_tau_nist = np.log(nist_tau)

# --------------------
# Plotting
# --------------------
plt.figure(figsize=(8, 6))
plt.errorbar(logZ_exp, log_tau_exp, yerr=log_tau_err, fmt='o', capsize=5, label='Experimental', color='blue')
plt.scatter(logZ_nist, log_tau_nist, color='orange', label='NIST')

# Annotate materials
for x, y, label in zip(logZ_exp, log_tau_exp, labels):
    plt.text(x, y + 0.1, label, ha='center', fontsize=10)

plt.xlabel(r'$\ln(Z)$')
plt.ylabel(r'$\ln(\tau / 10^{-24})$')
plt.title("Log-Log Plot of Attenuation Coefficient vs Atomic Number")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()
