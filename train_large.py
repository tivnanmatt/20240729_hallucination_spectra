import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from common_large import get_diffusion_bridge_model, load_weights, save_weights



# Get the model
diffusion_bridge_model = get_diffusion_bridge_model()

# Load pre-trained weights if available
weights_filename = 'weights/diffusion_backbone_weights_0728.pth'

# If weights are available, load them
if os.path.exists(weights_filename):
    load_weights(diffusion_bridge_model, weights_filename)

# Train the model
training_loss = diffusion_bridge_model.train_diffusion_backbone(batch_size=32, 
                                                num_epochs=100, 
                                                num_iterations_per_epoch=100,
                                                num_epochs_per_save=10,
                                                weights_filename=weights_filename,
                                                verbose=True)




plt.figure()
plt.plot(np.arange(len(training_loss)), training_loss)
plt.xlabel('Epoch')
plt.ylabel('Average Training Loss')
plt.title('Training Loss')
plt.savefig('figures/training_loss.png')