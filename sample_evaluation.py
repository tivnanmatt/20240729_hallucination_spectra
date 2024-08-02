import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from common_large import get_diffusion_bridge_model, load_weights, save_weights

num_images = 1024
num_measurements_per_image = 1
num_reconstructions_per_measurement = 1

num_timesteps = 256

sample_images = False

# Get the model
HU = np.sqrt(0.001695)/20
measurement_noise_variance = (100*HU)**2.0
diffusion_bridge_model = get_diffusion_bridge_model(measurement_noise_variance=measurement_noise_variance, train=False)

# Load pre-trained weights if available
weights_filename = 'weights/diffusion_backbone_weights_0728.pth'

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

        if sample_images:
            true_images[iImage,0,0] = diffusion_bridge_model.sample_images()
        else:
            true_images[iImage, 0, 0] = diffusion_bridge_model.image_dataset.images[iImage]
        

        for iMeasurement in range(num_measurements_per_image):
            measurements[iImage, iMeasurement,0] = diffusion_bridge_model.sample_measurements_given_images(true_images[iImage,0,0])

            for iReconstruction in range(num_reconstructions_per_measurement):   
                reconstructions[iImage, iMeasurement, iReconstruction] = diffusion_bridge_model.sample_reconstructions_given_measurements(measurements[iImage, iMeasurement,0].unsqueeze(0), num_timesteps=num_timesteps, verbose=False)[0][0]
                print(f'Image {iImage+1}/{num_images}, Measurement {iMeasurement+1}/{num_measurements_per_image}, Reconstruction {iReconstruction+1}/{num_reconstructions_per_measurement}')


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

# save true images, measurements, and reconstructions
true_images_filename = 'samples/20240801_LLM_observer/true_images.pt'
measurements_filename = 'samples/20240801_LLM_observer/measurements.pt'
reconstructions_filename = 'samples/20240801_LLM_observer/reconstructions.pt'

torch.save(true_images, true_images_filename)
torch.save(measurements, measurements_filename)
torch.save(reconstructions, reconstructions_filename)

print()