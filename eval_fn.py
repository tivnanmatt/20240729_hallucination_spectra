import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from common_large import get_diffusion_bridge_model, load_weights
from map_fn_new import *
from filter_fn import *
import numpy
import pandas as pd
import seaborn as sns

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

def display_difference(folder, image_sets, image_sets_d):
    true_images, measurements, reconstructions = image_sets
    true_images_d, measurements_d, reconstructions_d = image_sets_d

    mean, std = get_mean_std()
    difference = true_images_d - true_images
    difference = difference * std + mean
    display_map(difference, "Difference between true images and perturbed true images", 
                folder + "difference.png", plot_min, plot_max)
    
    return 0

def error_maps(folder, nums, image_sets):
    # mean over reconstructions
    mean = calculate_mean(nums, image_sets)
    
    # RMSE
    rmse = calculate_mse(nums, image_sets, freq=False)
    display_map(rmse, "RMSE maps", folder + "rmse.png", 0, 40)

    # Bias
    bias = calculate_bias(mean, nums, image_sets, freq=False)
    display_map(bias, "Bias maps", folder + "bias.png", 0, 40)

    # STD
    std = calculate_variance(mean, nums, image_sets, freq=False)
    display_map(std, "STD maps", folder + "std.png", 0, 40)

    # average across pixels and patients

    
    # filtered
    # recon_filtered_all = filter_recon(image_sets)
    # bandpass(folder, nums, image_sets, recon_filtered_all)
    # lowpass(folder, nums, image_sets, recon_filtered_all)

    return rmse, bias, std


def error_freq(folder, nums, image_sets):
    # unpack parameters
    true_images, measurements, reconstructions = image_sets

    # take 2D FFT for the frequency domain
    true_freq = torch.fft.fft2(true_images)
    recon_freq = torch.fft.fft2(reconstructions)
    image_sets = true_freq, measurements, recon_freq

    mean = calculate_mean(nums, image_sets)

    # MSE
    rmse = calculate_mse(nums, image_sets, freq=True)
    display_map(rmse, "RMSE maps (frequency)", folder + "rmse_freq.png", 4, 10)

    # Bias-squared
    bias = calculate_bias(mean, nums, image_sets, freq=True)
    display_map(bias, "Bias maps (frequency)", folder + "bias_freq.png", 4, 10)

    # Variance
    std = calculate_variance(mean, nums, image_sets, freq=True)
    display_map(std, "STD maps (frequency)", folder + "std_freq.png", 4, 10)
   
    return rmse, bias, std


def error_avg(error_maps):
    # average across pixels
    errors = avg_across_pixels(error_maps)
    # average across patients
    error, std = avg_across_patients(errors)

    return errors, error, std

def calculate_error(rmse, bias, std, frequency):
    if frequency:
        print("Frequency domain: ")
    rmse_vector, rmse_avg, rmse_std= error_avg(rmse)
    print(f"RMSE {round(rmse_avg[0].item(), 2)}" + u"\u00B1" f"{round(rmse_std[0].item(), 2)}")
    bias_vector, bias_avg, bias_std= error_avg(bias)
    print(f"BIAS {round(bias_avg[0].item(), 2)}" + u"\u00B1" f"{round(bias_std[0].item(), 2)}")
    std_vector, std_avg, std_std= error_avg(std)
    print(f"STD {round(std_avg[0].item(), 2)}" + u"\u00B1" f"{round(std_std[0].item(), 2)}")

    # the mean and std across all patients, single values only 
    all_errors = rmse_avg, rmse_std, bias_avg, bias_std, std_avg, std_std
    # length is the number of patients, one value for each patient in the vector
    error_vectors = rmse_vector, bias_vector, std_vector

    return all_errors, error_vectors

def plot_error(folder, all_errors, all_errors_digit, frequency=False):
    rmse_avg, rmse_std, bias_avg, bias_std, std_avg, std_std = all_errors
    rmse_avg_d, rmse_std_d, bias_avg_d, bias_std_d, std_avg_d, std_std_d = all_errors_digit
    if frequency:
        bar_plot_error("RMSE (frequency)", folder + "rmse_bar_f.png", rmse_avg, rmse_std, rmse_avg_d, rmse_std_d)
        bar_plot_error("Bias (frequency)", folder + "bias_bar_f.png", bias_avg, bias_std, bias_avg_d, bias_std_d)
        bar_plot_error("STD (frequency)", folder + "std_bar_f.png", std_avg, std_std, std_avg_d, std_std_d)

    else:
        bar_plot_error("RMSE", folder + "rmse_bar.png", rmse_avg, rmse_std, rmse_avg_d, rmse_std_d)
        bar_plot_error("Bias", folder + "bias_bar.png", bias_avg, bias_std, bias_avg_d, bias_std_d)
        bar_plot_error("STD", folder + "std_bar.png", std_avg, std_std, std_avg_d, std_std_d)

    return 0

def bar_plot_error(title, filename, error_1, std_1, error_2, std_2):
    groups = ["True Images", "True Images with Digits"]
    error = [error_1[0].item(), error_2[0].item()]
    std = [std_1[0].item(), std_2[0].item()]

    colors = ["darkgray", "dimgray"]

    fig, ax = plt.subplots(figsize=(4, 5))
    im = ax.bar(groups, error, yerr=std, color=colors)
    ax.bar_label(im, labels = [f"{round(error[0], 2)}" + u"\u00B1" f"{round(std[0], 2)}", 
                               f"{round(error[1], 2)}" + u"\u00B1" f"{round(std[1], 2)}"])
    ax.set_title(title, weight="bold")
    plt.show()
    plt.savefig(filename)

    return 0
"""
def violin_plot_error(error_vectors, error_vectors_f, error_vectors_d, error_vectors_d_f):
    # all of these are vectors (length is the number of patients)
    rmse, bias, std = error_vectors
    rmse_f, bias_f, std_f = error_vectors_f
    rmse_d, bias_d, std_d = error_vectors_d
    rmse_d_f, bias_d_f, std_d_f = error_vectors_d_f

    # tensors to numpy arrays
    rmse = torch.squeeze(rmse).numpy()
    rmse_d = torch.squeeze(rmse_d).numpy()
    rmse_f = torch.squeeze(rmse_f).numpy()
    rmse_d_f = torch.squeeze(rmse_d_f).numpy() 

    bias = torch.squeeze(bias).numpy()
    bias_d = torch.squeeze(bias_d).numpy()
    bias_f = torch.squeeze(bias_f).numpy()
    bias_d_f = torch.squeeze(bias_d_f).numpy()

    std = torch.squeeze(std).numpy()
    std_d = torch.squeeze(std_d).numpy()
    std_f = torch.squeeze(std_f).numpy()
    std_d_f = torch.squeeze(std_d_f).numpy()

    # concatenate arrays, put that into df
    # need to check the shape of the error vectors, determine which direction to concatenate

    # numpy arrays to dataframe
    header = pd.MultiIndex.from_product([['RMSE', 'RMSE (frequency)', 'Bias', 'Bias (frequency)', 'STD', 'STD (frequency)'], 
                                         ['True images', 'True images with digits']],
                                         names=['error type', 'true type'])
    df = pd.DataFrame(data, columns=header)

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.violinplot(data=df, x='error type', y=, hue='true type', split='true', inner='box', 
                   palette={"True images": "darkgray", "True image with digits": "dimgray"})
    fig.suptitle("Evaluation Error", fontweight='bold')
    ax.set_xlabel('error type')
    ax.set_ylabel('Error')

    return 0
"""
    
# output mean and std of errors
def record_errors(all_errors, filename, frequency=False, perturbation=False):
    rmse_avg, rmse_std, bias_avg, bias_std, std_avg, std_std = all_errors
    file = open(filename, 'a')
    rmse = f"RMSE {round(rmse_avg[0].item(), 2)}" + u"\u00B1" f"{round(rmse_std[0].item(), 2)}\n"
    bias = f"BIAS {round(bias_avg[0].item(), 2)}" + u"\u00B1" f"{round(bias_std[0].item(), 2)}\n"
    std = f"STD {round(std_avg[0].item(), 2)}" + u"\u00B1" f"{round(std_std[0].item(), 2)}\n"

    if frequency:
        file.write("Frequency domain. ")
    if perturbation:
        file.write("With digits. ")

    file.write("\n")
    file.write(rmse)
    file.write(bias)
    file.write(std)
    file.write("\n")

    file.close()

    return 0

def create_animation(folder, file, nums, image_sets):
    true_images, measurements, reconstructions = image_sets
    # num_pixel, num_images, num_measurements, num_reconstructions = nums
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axs[0].imshow(true_images[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=-2, vmax=2)
    axs[0].set_title('True Images')
    im1 = axs[1].imshow(measurements[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=-2, vmax=2)
    axs[1].set_title('Measurements')
    im2 = axs[2].imshow(reconstructions[0, 0, 0, 0].detach().cpu(), cmap='gray', vmin=-2, vmax=2)
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

"""
check histogram
# print("MSE", mse)
# hist_mse, edge_mse = torch.histogram(mse[0, 0, 0, :, :, :], bins=100)
# print(hist_mse, edge_mse)
# fig = plt.plot(edge_mse[:-1], hist_mse)
# plt.title("mse")
# plt.xlim(0, 50)
# plt.show()
# plt.savefig("maps_a/mse_hist.png")

"""