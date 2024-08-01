import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from common_large import get_diffusion_bridge_model, load_weights, save_weights

# Get the model
HU = np.sqrt(0.001695)/20
measurement_noise_variance = (20*HU)**2.0
diffusion_bridge_model = get_diffusion_bridge_model(measurement_noise_variance=measurement_noise_variance)

# Load pre-trained weights if available
weights_filename = 'weights/diffusion_backbone_weights_20HU_0801.pth'

# If weights are available, load them
if os.path.exists(weights_filename):
    load_weights(diffusion_bridge_model, weights_filename)

# Train the model
# 128, 1000, 200, 10
training_loss = diffusion_bridge_model.train_diffusion_backbone(batch_size=128, 
                                                num_epochs=1000, 
                                                num_iterations_per_epoch=200,
                                                num_epochs_per_save=10,
                                                weights_filename=weights_filename,
                                                verbose=True)

plt.figure(figsize=(10, 5))
plt.plot(np.arange(len(training_loss)), training_loss)
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.title('Training Loss')
plt.savefig('figures/training_loss.png')

np.save('figures/training_loss.npy', training_loss)