import os


# set cuda visible devices to 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
from common_large import get_diffusion_bridge_model, load_weights, save_weights

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_images = 1
num_measurements_per_image = 1
num_reconstructions_per_measurement = 1

num_timesteps = 8

sample_images = True

# Get the model
HU = np.sqrt(0.001695)/20
measurement_noise_variance = (100*HU)**2.0
diffusion_bridge_model = get_diffusion_bridge_model(measurement_noise_variance=measurement_noise_variance, train=False, num_files=1)

# Load pre-trained weights if available
weights_filename = 'weights/diffusion_backbone_weights_100HU.pth'
# weights_filename = 'weights/diffusion_backbone_weights_100HU_custom.pth'

# If weights are available, load them
if os.path.exists(weights_filename):
    load_weights(diffusion_bridge_model, weights_filename)
else:
    raise ValueError(f'Weights file {weights_filename} does not exist')

diffusion_bridge_model.eval()

timesteps = torch.linspace(1, 0, num_timesteps+1).to(device)**2.0
timesteps = torch.linspace(1, 0, num_timesteps+1).to(device)






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
                reconstructions[iImage, iMeasurement, iReconstruction] = diffusion_bridge_model.sample_reconstructions_given_measurements(measurements[iImage, iMeasurement,0].unsqueeze(0), timesteps=timesteps, verbose=True)[0][0]
                
                # t = torch.ones((1,1), device=device)
                # reconstructions[iImage, iMeasurement, iReconstruction] = diffusion_bridge_model.image_reconstructor.diffusion_model.sample_x_t_given_x_0(true_images[iImage, iMeasurement,0], t)
                # reconstructions[iImage, iMeasurement, iReconstruction] = diffusion_bridge_model.image_reconstructor.diffusion_model.predict_x_0_given_x_t(reconstructions[iImage, iMeasurement, iReconstruction].unsqueeze(0), t)[0] # reverse prediction
                
                print(f'Image {iImage+1}/{num_images}, Measurement {iMeasurement+1}/{num_measurements_per_image}, Reconstruction {iReconstruction+1}/{num_reconstructions_per_measurement}')

# print the mean squared error for both measurements and reconstructions
print(f'Mean Squared Error for Measurements: {torch.mean((true_images - measurements)**2.0):.4f}')
print(f'Mean Squared Error for Reconstructions: {torch.mean((true_images - reconstructions)**2.0):.4f}')


# Convert standard units to Hounsfield Units (HU)
def standard_to_hu(tensor, mu=-572.3447, sigma=487.3876):
    return tensor * sigma + mu

# Rescale to (0, 1) using abdomen window and clip values outside the range
def rescale_abdomen_window(tensor, window_center=40, window_width=400):
    min_hu = window_center - (window_width / 2)
    max_hu = window_center + (window_width / 2)
    tensor = (tensor - min_hu) / (max_hu - min_hu)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor

# soft_tissue_mask = true_images < 1.5

true_images = rescale_abdomen_window(standard_to_hu(true_images))
measurements = rescale_abdomen_window(standard_to_hu(measurements))
reconstructions = rescale_abdomen_window(standard_to_hu(reconstructions))

# true_images[soft_tissue_mask] *= 2.0
# true_images = soft_tissue_mask

fig = plt.figure(figsize=(5, 5))
vmin = 0
vmax = 1
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