import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from common_large import get_diffusion_bridge_model, load_weights
# import laboratory_tcga as lab
from map_fn_new import *
from filter_fn import *

plot_min = -160
plot_max = 240
"""
# nums = num_pixel, num_images, num_measurements, num_reconstructions
# set up the model in the evaluation mode and generate samples
def setup_sampling(nums):
    # Get the model
    diffusion_bridge_model = get_diffusion_bridge_model(train=False)

    # Load pre-trained weights
    diffusion_backbone_weights_filename = 'weights/diffusion_backbone_weights_20HU_0801.pth'
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
"""
    
def check_measurement_var(nums, image_sets):
    true_images, measurements, reconstructions = image_sets
    measurements_var = torch.var(measurements, axis=1, keepdim=True)
    print("average measurements variation:", torch.mean(measurements_var))

    return measurements_var

def display_image_sets(folder, image_sets):
    true_images, measurements, reconstructions = image_sets

    # reverse the normalization to HU
    mean, std = get_mean_std()
    true_images = true_images * std  + mean
    measurements = measurements * std + mean
    reconstructions = reconstructions * std + mean

    display_map(true_images, "True images", folder + "true.png", plot_min, plot_max)
    display_map(measurements, "Measurements", folder + "measurements.png", plot_min, plot_max)
    display_map(reconstructions, "Reconstructions", folder + "reconstructions.png", plot_min, plot_max)

    return 0

def error_maps(folder, nums, image_sets):
    # mean over reconstructions
    mean = calculate_mean(nums, image_sets)

    # MSE
    mse = calculate_mse(nums, image_sets, freq=False)
    display_map(mse, "MSE maps", folder + "mse.png", plot_min, plot_max)

    # Bias-squared
    bias = calculate_bias(mean, nums, image_sets, freq=False)
    display_map(bias, "Bias-squared maps", folder + "bias.png", plot_min, plot_max)

    # Variance
    variance = calculate_variance(mean, nums, image_sets, freq=False)
    display_map(variance, "Variance maps", folder + "variance.png", plot_min, plot_max)

    # filtered
    recon_filtered_all = filter_recon(image_sets)
    bandpass(folder, nums, image_sets, recon_filtered_all)
    lowpass(folder, nums, image_sets, recon_filtered_all)

    return 0


def error_freq(folder, nums, image_sets):
    # unpack parameters
    true_images, measurements, reconstructions = image_sets

    # take 2D FFT for the frequency domain
    true_freq = torch.fft.fft2(true_images)
    recon_freq = torch.fft.fft2(reconstructions)
    image_sets = true_freq, measurements, recon_freq

    mean = calculate_mean(nums, image_sets)

    # MSE
    mse = calculate_mse(nums, image_sets, freq=False)
    display_map(mse, "MSE maps (frequency)", folder + "mse_freq.png", plot_min, plot_max)

    # Bias-squared
    bias = calculate_bias(mean, nums, image_sets, freq=False)
    display_map(bias, "Bias-squared maps (frequency)", folder + "bias_freq.png", plot_min, plot_max)

    # Variance
    variance = calculate_variance(mean, nums, image_sets, freq=False)
    display_map(variance, "Variance maps (frequency)", folder + "variance_freq.png", plot_min, plot_max)

    return 0


def create_animation(folder, file, nums, image_sets):
    true_images, measurements, reconstructions = image_sets
    # num_pixel, num_images, num_measurements, num_reconstructions = nums
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axs[0].imshow(true_images[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=-1.2, vmax=1.2)
    axs[0].set_title('True Images')
    im1 = axs[1].imshow(measurements[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=-1.2, vmax=1.2)
    axs[1].set_title('Measurements')
    im2 = axs[2].imshow(reconstructions[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=-1.2, vmax=1.2)
    axs[2].set_title('Reconstructions')

    def animate(i):
        print('Animating frame {}/{}'.format(i+1, num_images*num_measurements*num_reconstructions))
        i, j, k = i // (num_measurements*num_reconstructions), (i // num_reconstructions) % num_measurements, i % num_reconstructions
        im0.set_array(true_images[i, 0, 0, 0].detach().cpu())
        im1.set_array(measurements[i, j, 0, 0].detach().cpu())
        im2.set_array(reconstructions[i, j, k, 0].detach().cpu())
        return im0, im1, im2
        

    ani = animation.FuncAnimation(fig, animate, frames=num_images*num_measurements*num_reconstructions, interval=1000, repeat=False)

    # mp4 writer ffmpeg
    writer = animation.writers['ffmpeg'](fps=10)
    ani.save(folder + file, writer=writer)

    plt.show()

    return 0

