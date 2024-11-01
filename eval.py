import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from common_large import get_diffusion_bridge_model, load_weights, save_weights

num_images = 16
num_measurements_per_image = 16
num_reconstructions_per_measurement = 1

num_timesteps = 1024

def sample_recon(nums, noise_hu):
    num_images, num_measurements_per_image, num_reconstructions_per_measurement, num_pixels, num_timesteps = nums

    # Get the model
    HU = np.sqrt(0.001695)/20
    measurement_noise_variance = (noise_hu*HU)**2.0
    diffusion_bridge_model = get_diffusion_bridge_model(measurement_noise_variance=measurement_noise_variance, train=False)

    # Load pre-trained weights if available
    weights_filename = 'weights/diffusion_backbone_weights_20HU_0801.pth'

    # If weights are available, load them
    if os.path.exists(weights_filename):
        load_weights(diffusion_bridge_model, weights_filename)

    diffusion_bridge_model.eval()

    with torch.no_grad():
        true_image, measurements, reconstructions = diffusion_bridge_model.sample_reconstructions()

    image_shape = true_image[0].shape
    measurement_shape = measurements[0].shape
    reconstruction_shape = reconstructions[0].shape

    true_images = torch.zeros((num_images, 1, 1, *image_shape), dtype=true_image.dtype, device=true_image.device)
    measurements = torch.zeros((num_images, num_measurements_per_image, 1, *measurement_shape), dtype=measurements.dtype, device=measurements.device)
    reconstructions = torch.zeros((num_images, num_measurements_per_image, num_reconstructions_per_measurement, *reconstruction_shape), dtype=reconstructions.dtype, device=reconstructions.device)

    with torch.no_grad():                 
        for iImage in range(num_images):
            true_images[iImage,0,0] = diffusion_bridge_model.sample_images()

            for iMeasurement in range(num_measurements_per_image):
                measurements[iImage, iMeasurement,0] = diffusion_bridge_model.sample_measurements_given_images(true_images[iImage,0,0])

                for iReconstruction in range(num_reconstructions_per_measurement):   
                    reconstructions[iImage, iMeasurement, iReconstruction] = diffusion_bridge_model.sample_reconstructions_given_measurements(measurements[iImage, iMeasurement,0].unsqueeze(0), num_timesteps=num_timesteps, verbose=True)[0][0]
    
    image_sets = true_images, measurements, reconstructions

    return image_sets

"""
fig = plt.figure(figsize=(5, 5))
vmin = -2
vmax = 2
plt.imshow(true_images[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.title('True Image')
plt.savefig('figures/true_image.png')

fig = plt.figure(figsize=(5, 5))
plt.imshow(measurements[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.title('Measurement')
plt.savefig('figures/measurement.png')

fig = plt.figure(figsize=(5, 5))
plt.imshow(reconstructions[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
plt.axis('off')
plt.title('Reconstruction')
plt.savefig('figures/reconstruction.png')


print()

"""