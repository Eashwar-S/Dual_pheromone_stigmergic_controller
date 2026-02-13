import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Data (Search with Failures)
# ---------------------------

grid_size = np.array([200, 300, 400, 500])

# Diverge
diverge_avg = np.array([663.2, 822.6, 1353.4, 2655.2])
diverge_std = np.array([237.2155138, 251.9212178, 272.3752926, 1521.622555])

# Stigmergy Random Walk
stigmergy_avg = np.array([1563, 2337.4, 4916, 4341.8])
stigmergy_std = np.array([902.4527688, 365.1524339, 2599.818744, 2326.050236])

# Centralized
centralized_avg = np.array([931, 3023, 5154, 7082])

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(9, 6))

# Diverge
plt.errorbar(grid_size, diverge_avg, yerr=diverge_std, fmt='o--', capsize=5,
             label='Diverge', linewidth=2, color='blue')

# Stigmergy Random Walk
plt.errorbar(grid_size, stigmergy_avg, yerr=stigmergy_std, fmt='s--', capsize=5,
             label='Stigmergy Random Walk', linewidth=2, color='green')

# Centralized
plt.plot(grid_size, centralized_avg, 'd--', label='Centralized', linewidth=2, color='orange')

# ---------------------------
# Label mean ± std
# ---------------------------
for x, mean, std in zip(grid_size, diverge_avg, diverge_std):
    plt.text(x, mean + std + 100, f'{mean:.1f}±{std:.1f}', ha='center', fontsize=9, color='blue')

for x, mean, std in zip(grid_size, stigmergy_avg, stigmergy_std):
    plt.text(x, mean + std + 100, f'{mean:.1f}±{std:.1f}', ha='center', fontsize=9, color='green')

for x, mean in zip(grid_size, centralized_avg):
    plt.text(x, mean + 100, f'{mean:.1f}', ha='center', fontsize=9, color='orange')

# ---------------------------
# Formatting
# ---------------------------
plt.xlabel('Grid Size', fontsize=12)
plt.ylabel('Average Steps to Converge', fontsize=12)
plt.title('Comparison of Search Strategies (with Failures)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
