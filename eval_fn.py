import torch
import matplotlib.pyplot as plt
from common_large import get_diffusion_bridge_model, load_weights
import laboratory_tcga as lab
from map_fn_new import *
# from filter_functions import *

plot_min = -160
plot_max = 240

# nums = num_pixel, num_images, num_measurements, num_reconstructions
# set up the model in the evaluation mode and generate samples
def setup_sampling(nums):
    # Get the model
    diffusion_bridge_model = get_diffusion_bridge_model(train=False)

    # Load pre-trained weights
    diffusion_backbone_weights_filename = 'weights/diffusion_backbone_weights_0728.pth'
    load_weights(diffusion_bridge_model, diffusion_backbone_weights_filename)

    # Set the model to evaluation mode
    diffusion_bridge_model.eval()

    # Generate samples
    # unpack parameters
    num_pixel, num_images, num_measurements, num_reconstructions = nums
    true_images = torch.zeros(num_images, 1, 1, 1, num_pixel, num_pixel)
    measurements = torch.zeros(num_images, num_measurements, 1, 1, num_pixel, num_pixel)
    reconstructions = torch.zeros(num_images, num_measurements, num_reconstructions, 1, num_pixel, num_pixel)

    for i in range(num_images):
        # getting true images
        true_image = diffusion_bridge_model.image_dataset[i:i+1]
        true_images[i] = true_image.reshape(1, 1, 1, num_pixel, num_pixel)
        # reconstructing
        for j in range(num_measurements):
            measurement = diffusion_bridge_model.measurement_simulator(true_image)
            measurements[i, j] = measurement
            for k in range(num_reconstructions):
                print(f"Generating sample {i+1}/{num_images}, measurement {j+1}/{num_measurements}, reconstruction {k+1}/{num_reconstructions}")
                with torch.no_grad():
                    assert isinstance(diffusion_bridge_model.image_reconstructor, lab.torch.tasks.reconstruction.DiffusionBridgeImageReconstructor)
                    assert isinstance(diffusion_bridge_model.image_reconstructor.diffusion_model, lab.torch.diffusion.UnconditionalDiffusionModel)
                    reconstruction = diffusion_bridge_model.image_reconstructor(measurement, num_timesteps=32)
                reconstructions[i, j, k] = reconstruction.reshape(1, num_pixel, num_pixel)
    
    image_sets = true_images, measurements, reconstructions

    return image_sets

def check_measurement_var(nums, image_sets):
    true_images, measurements, reconstructions = image_sets
    # num_pixel, num_images, num_measurements, num_reconstructions = nums
    # mean_measu = calculate_mean(num_images, num_measurements, num_pixel, measurements)
    # variance_measurement = calculate_variance(num_images, num_pixel, mean_measu, num_measurements, measurements)
    # display_variance_measurement(variance_measurement, plot_min, plot_max, freq=False, bandpass=False, lowpass=False, u=0)
    # print("variance_measurement:", variance_measurement)
    measurements_var = torch.var(measurements, axis=1, keepdim=True)
    # print("measurements variation:", measurements_var)
    print("average measurements variation:", torch.mean(measurements_var))

    # return variance_measurement
    return measurements_var

def display_image_sets(image_sets):
    true_images, measurements, reconstructions = image_sets
    display_true(true_images, 0)
    display_true(measurements, 1)
    display_true(reconstructions, 2)

    return 0


def error_maps(nums, image_sets):
    # unpack parameters
    num_pixel, num_images, num_measurements, num_reconstructions = nums
    true_images, measurements, reconstructions = image_sets
    mean_recon = calculate_mean(num_images, num_reconstructions, num_pixel, reconstructions)

    # MSE
    mse = calculate_mse(num_images, num_reconstructions, num_pixel, true_images, reconstructions)
    display_mse(mse, plot_min, plot_max, freq=False)

    # Bias-squared
    bias = calculate_bias(num_images, num_pixel, mean_recon, true_images)
    display_bias(bias, plot_min, plot_max, freq=False)

    # Variance
    variance = calculate_variance(num_images, num_pixel, mean_recon, num_reconstructions, reconstructions)
    display_variance(variance, plot_min, plot_max, freq=False, bandpass=False, lowpass=False, u=0)

    return 0



"""
# True images
display_true(true_images)
# display_true(measurements)
# display_true(reconstructions)

### frequency domain
# Convert true_images and reconstructions to their spatial frequency domain
true_freq = torch.fft.fft2(true_images)
recon_freq = torch.fft.fft2(reconstructions)

# frequency MSE
mse_freq = calculate_mse(num_images, num_reconstructions, num_pixel, true_freq, recon_freq)
log_mse_freq = torch.log(mse_freq)
display_mse(log_mse_freq, plot_min, plot_max, freq=True)

freq_mean = calculate_mean(num_images, num_reconstructions, num_pixel, recon_freq)

# frequency Bias-squared
bias_freq = calculate_bias(num_images, num_pixel, freq_mean, true_freq)
log_bias_freq = torch.log(bias_freq)
display_bias(log_bias_freq, plot_min, plot_max, freq=True)

# frequency Variance
var_freq = calculate_variance(num_images, num_pixel, freq_mean, num_reconstructions, recon_freq)
log_var_freq = torch.log(var_freq)
display_variance(log_var_freq, plot_min, plot_max, freq=True, bandpass=False, lowpass=False, u=0)
"""

"""
### filtered
recon_filtered_all = apply_filter(reconstructions)
# bandpass variance
bandpass_variance(num_images, num_reconstructions, num_pixel, recon_filtered_all)
# lowpass variance
lowpass_variance(num_images, num_reconstructions, num_pixel, recon_filtered_all)

# Create animation
import matplotlib.animation as animation

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im0 = axs[0].imshow(true_images[0, 0, 0, 0], cmap='gray', vmin=-1.2, vmax=1.2)
axs[0].set_title('True Images')
im1 = axs[1].imshow(measurements[0, 0, 0, 0], cmap='gray', vmin=-1.2, vmax=1.2)
axs[1].set_title('Measurements')
im2 = axs[2].imshow(reconstructions[0, 0, 0, 0], cmap='gray', vmin=-1.2, vmax=1.2)
axs[2].set_title('Reconstructions')

def animate(i):
    print('Animating frame {}/{}'.format(i+1, num_images*num_measurements*num_reconstructions))
    i, j, k = i // (num_measurements*num_reconstructions), (i // num_reconstructions) % num_measurements, i % num_reconstructions
    im0.set_array(true_images[i, 0, 0, 0])
    im1.set_array(measurements[i, j, 0, 0])
    im2.set_array(reconstructions[i, j, k, 0])
    return im0, im1, im2
    

ani = animation.FuncAnimation(fig, animate, frames=num_images*num_measurements*num_reconstructions, interval=1000, repeat=False)

# mp4 writer ffmpeg
writer = animation.writers['ffmpeg'](fps=10)
ani.save('figures/diffusion_bridge_model_large_0722.mp4', writer=writer)

plt.show()
"""
