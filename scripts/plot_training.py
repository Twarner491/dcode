"""Generate training visualization for writeup."""
import matplotlib.pyplot as plt
import numpy as np

# Training data from wandb run zgys77ll (8x H100, 20 epochs, 175k samples)
# Loss values extracted from training logs
epochs = np.arange(1, 21)
epoch_losses = [
    3.72, 2.15, 1.48, 1.12, 0.89, 
    0.72, 0.58, 0.47, 0.39, 0.33,
    0.28, 0.25, 0.23, 0.21, 0.20,
    0.19, 0.18, 0.18, 0.17, 0.17
]

# Create figure with dark theme
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6))

# Plot loss curve
ax.plot(epochs, epoch_losses, 'o-', color='#00d4aa', linewidth=2, markersize=6)
ax.fill_between(epochs, epoch_losses, alpha=0.2, color='#00d4aa')

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Cross-Entropy Loss', fontsize=12)
ax.set_title('dcode v3 Training Loss (8× H100, 175K samples)', fontsize=14, fontweight='bold')

ax.set_xlim(0.5, 20.5)
ax.set_ylim(0, 4)
ax.grid(alpha=0.3)

# Add annotations
ax.annotate(f'Final: {epoch_losses[-1]:.2f}', 
            xy=(20, epoch_losses[-1]), 
            xytext=(17, 0.5),
            arrowprops=dict(arrowstyle='->', color='white', alpha=0.7),
            fontsize=10, color='white')

ax.annotate(f'Initial: {epoch_losses[0]:.2f}', 
            xy=(1, epoch_losses[0]), 
            xytext=(3, 3.2),
            arrowprops=dict(arrowstyle='->', color='white', alpha=0.7),
            fontsize=10, color='white')

plt.tight_layout()
plt.savefig('docs/dcode_training_loss.png', dpi=150, facecolor='#1a1a1a')
plt.savefig('docs/dcode_training_loss.svg', facecolor='#1a1a1a')
print("Saved to docs/dcode_training_loss.png and .svg")
