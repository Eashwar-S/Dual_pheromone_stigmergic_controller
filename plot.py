import matplotlib.pyplot as plt
import numpy as np

# Data
grid_size = np.array([200, 300, 400, 500])

# Diverge
diverge_avg = np.array([621.8, 1269.6, 1711.2, 2144.8])
diverge_std = np.array([303.1422109, 199.4299376, 369.8008924, 272.4953211])

# Stigmergy Random Walk
stigmergy_avg = np.array([794.2, 2440, 3402.2, 4234.4])
stigmergy_std = np.array([356.7642078, 834.0278772, 998.0354202, 773.7033023])

# Centralized
centralized_avg = np.array([1512, 2686, 3322, 3118])

# Plot
plt.figure(figsize=(9, 6))
plt.errorbar(grid_size, diverge_avg, yerr=diverge_std, fmt='o--', capsize=5, label='Diverge', linewidth=2)
plt.errorbar(grid_size, stigmergy_avg, yerr=stigmergy_std, fmt='s--', capsize=5, label='Stigmergy Random Walk', linewidth=2)
plt.plot(grid_size, centralized_avg, 'd--', label='Centralized', linewidth=2)

# Label mean ± std for each point
for x, mean, std in zip(grid_size, diverge_avg, diverge_std):
    plt.text(x, mean + std + 100, f'{mean:.1f}±{std:.1f}', ha='center', fontsize=9, color='blue')

for x, mean, std in zip(grid_size, stigmergy_avg, stigmergy_std):
    plt.text(x, mean + std + 100, f'{mean:.1f}±{std:.1f}', ha='center', fontsize=9, color='green')

for x, mean in zip(grid_size, centralized_avg):
    plt.text(x, mean + 100, f'{mean:.1f}', ha='center', fontsize=9, color='orange')

# Labels and Title
plt.xlabel('Grid Size', fontsize=12)
plt.ylabel('Average Steps to Converge', fontsize=12)
plt.title('Comparison of Search Strategies', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
