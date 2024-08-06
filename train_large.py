import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from common_large import get_diffusion_bridge_model, load_weights, save_weights

# Get the model
HU = np.sqrt(0.001695)/20
measurement_noise_variance = (100*HU)**2.0
diffusion_bridge_model = get_diffusion_bridge_model(measurement_noise_variance=measurement_noise_variance, train=True, num_files=1)

# Load pre-trained weights if available
weights_filename = 'weights/diffusion_backbone_weights_100HU.pth'
# weights_filename = 'weights/diffusion_backbone_weights_100HU_custom.pth'

# If weights are available, load them
if os.path.exists(weights_filename):
    load_weights(diffusion_bridge_model, weights_filename)

# If optimizer is available, load it
optimizer = torch.optim.Adam(diffusion_bridge_model.image_reconstructor.diffusion_model.diffusion_backbone.parameters(), lr=2e-4)
optimizer_filename = weights_filename.replace('.pth', '_optimizer.pth')
if os.path.exists(optimizer_filename):
    optimizer.load_state_dict(torch.load(optimizer_filename))

nRepeats = 1000
for iRepeat in range(nRepeats):
    training_loss = diffusion_bridge_model.train_diffusion_backbone(batch_size=4, 
                                                    num_epochs=20, 
                                                    num_iterations_per_epoch=100,
                                                    num_epochs_per_save=10,
                                                    weights_filename=weights_filename,
                                                    optimizer=optimizer,
                                                    ema=True,
                                                    verbose=True)

    # save the optimizer
    torch.save(optimizer.state_dict(), optimizer_filename)

    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(training_loss)), training_loss)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Average Training Loss')
    plt.title('Training Loss')
    plt.savefig('figures/training_loss.png')

    np.save('figures/training_loss.npy', training_loss)