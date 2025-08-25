import pickle
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to your results pickle file (update if needed)
pickle_file = '/home/exouser/multiloss-bfe/multiloss_pdbbind/PGGCN_results_100structures_20250825_034719.pkl'

with open(pickle_file, 'rb') as f:
    results_data = pickle.load(f)

all_results = results_data['all_results']

# Extract sweep data
def get_sweep_arrays(all_results):
    physics_weights = []
    final_train_losses = []
    test_losses = []
    maes = []
    epochs = []
    for res in all_results:
        physics_weights.append(res['physics_weight'])
        final_train_losses.append(res['final_train_loss'])
        test_losses.append(res['test_loss'])
        maes.append(res['mean_abs_diff'])
        epochs.append(res['epochs_trained'])
    return np.array(physics_weights), np.array(final_train_losses), np.array(test_losses), np.array(maes), np.array(epochs)

physics_weights, final_train_losses, test_losses, maes, epochs = get_sweep_arrays(all_results)

# 1. Loss/MAE vs. Physics Weight
plt.figure(figsize=(10,6))
plt.plot(physics_weights, final_train_losses, 'o-', label='Final Train Loss')
plt.plot(physics_weights, test_losses, 's-', label='Test Loss')
plt.plot(physics_weights, maes, '^-', label='MAE')
plt.xscale('log')
plt.xlabel('Physics Weight')
plt.ylabel('Loss / MAE')
plt.title('Loss and MAE vs. Physics Weight')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_mae_vs_physics_weight.png', dpi=200)

# 2. Train vs Test Loss Scatter
plt.figure(figsize=(6,6))
plt.scatter(final_train_losses, test_losses, c=physics_weights, cmap='viridis', s=80)
for i, pw in enumerate(physics_weights):
    plt.text(final_train_losses[i], test_losses[i], f'{pw:.0e}', fontsize=8)
plt.xlabel('Final Train Loss')
plt.ylabel('Test Loss')
plt.title('Train vs Test Loss (color: physics_weight)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('train_vs_test_loss_scatter.png', dpi=200)

# 3. MAE vs Total Loss Scatter
plt.figure(figsize=(6,6))
plt.scatter(maes, test_losses, c=physics_weights, cmap='plasma', s=80)
for i, pw in enumerate(physics_weights):
    plt.text(maes[i], test_losses[i], f'{pw:.0e}', fontsize=8)
plt.xlabel('MAE')
plt.ylabel('Test Loss')
plt.title('MAE vs Test Loss (color: physics_weight)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('mae_vs_test_loss_scatter.png', dpi=200)

# 4. Epoch-wise Loss Curves for Each Sweep
plt.figure(figsize=(10,6))
for res in all_results:
    losses = res['training_history']['total_losses']
    plt.plot(np.arange(1, len(losses)+1), losses, label=f'pw={res["physics_weight"]:.0e}')
plt.xlabel('Epoch')
plt.ylabel('Total Loss')
plt.title('Epoch-wise Loss Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('epochwise_loss_curves.png', dpi=200)

# 5. Train vs Test Scatter for Each Sweep
for res in all_results:
    pw = res['physics_weight']
    y_true_test = res['y_true_test']
    y_pred_test = res['y_pred_test']
    y_true_train = res['y_true_train']
    y_pred_train = res['y_pred_train']
    plt.figure(figsize=(7,7))
    plt.scatter(y_true_train, y_pred_train, alpha=0.6, label='Train', color='blue')
    plt.scatter(y_true_test, y_pred_test, alpha=0.6, label='Test', color='red')
    plt.plot([min(y_true_train), max(y_true_train)], [min(y_true_train), max(y_true_train)], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Train vs Test Scatter (physics_weight={pw:.0e})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'train_vs_test_scatter_pw_{pw:.0e}.png', dpi=200)
    plt.close()

print("All plots generated and saved.")
