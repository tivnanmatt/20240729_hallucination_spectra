"""
map_functions.py: Contains functions to calculate and display the MSE, bias-squared, and variance maps of 
reconstructions. All maps are displayed in grayscale. 
"""
import torch
import matplotlib.pyplot as plt
import laboratory as lab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = lab.torch.datasets.TCGA(
                            root='TCGA_LIHC',
                            train=False).to(device)
mean = loader.mu
print("loader mean:", mean)
std = loader.sigma
print("loader std:", std)

def get_mean_std():
    return mean, std

# Find the mean across reconstructions
def calculate_mean(nums, image_sets):
    # mean = torch.mean(reconstructions, 2, keepdim=True)
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    mean = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels, dtype=torch.complex64)
    # for every sample image
    for n in range(num_images):
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                mean[n, :, :, :, :, :] += reconstructions[n, m, r, :, :, :].detach().cpu()
        # mean of this sample image
        mean[n, :, :, :, :, :] /= (num_measurements * num_reconstructions)

    return mean

# MSE
def calculate_mse(nums, image_sets, freq=False):
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    mse = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels)
    for n in range(num_images):
        total_diff = torch.zeros(1, 1, 1, 1, num_pixels, num_pixels)
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                # find the difference between each reconstruction and the ground truth
                # diff_squared, total_diff, mse are all tensors of the same shape as the images
                diff_squared = (true_images[n, :, :, :, :, :].detach().cpu() - reconstructions[n, m, r, :, :, :].detach().cpu()).abs() ** 2
                total_diff += diff_squared

        # MSE for this ground truth image
        mse[n, :, :, :, :, :] = total_diff / (num_measurements * num_reconstructions)
        # RMSE
        mse[n, :, :, :, :, :] = torch.sqrt(mse[n, :, :, :, :, :])
        # reverse the normalization to HU
        mse[n, :, :, :, :, :] = mse[n, :, :, :, :, :] * std
    
    if freq:
        log_mse = torch.log(mse)
        return log_mse
    else:
        return mse


# Bias-squared 
def calculate_bias(mean, nums, image_sets, freq=False):
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    bias = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels)
    for m in range(num_images):
        # bias[m, :, :, :, :, :] = (mean[m, :, :, :, :, :] - true_images[m, :, :, :, :, :].detach().cpu()).abs() ** 2
        bias[m, :, :, :, :, :] = (mean[m, :, :, :, :, :] - true_images[m, :, :, :, :, :].detach().cpu()).abs()
        # reverse normalization
        # bias[m, :, :, :, :, :] = bias[m, :, :, :, :, :] * (std ** 2)
        bias[m, :, :, :, :, :] = bias[m, :, :, :, :, :] * std

    if freq:
        log_bias = torch.log(bias)
        return log_bias
    else:
        return bias


# Variance
def calculate_variance(mean, nums, image_sets, freq=False):
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    variance = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels)
    for n in range(num_images):
        # total_var = torch.zeros(1, 1, 1, 1, num_pixels, num_pixels)
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                # total_var[:, :, :, :, :, :] += (reconstructions[n, m, r, :, :, :].detach().cpu() - mean[n, :, :, :, :, :]).abs() ** 2
                variance[n, :, :, :, :, :] += (reconstructions[n, m, r, :, :, :].detach().cpu() - mean[n, :, :, :, :, :]).abs() ** 2
        # variance among the reconstructions of this sample image
        if (num_measurements * num_reconstructions - 1) == 0:
            divisor = 1
        else:
            divisor = (num_measurements * num_reconstructions - 1)
        variance[n, :, :, :, :, :] = torch.div(variance[n, :, :, :, :, :], divisor) 
        # use std
        variance[n, :, :, :, :, :] = torch.sqrt(variance[n, :, :, :, :, :])
        # reverse the normalization 
        # variance[n, :, :, :, :, :] = variance[n, :, :, :, :, :] * (std ** 2)
        variance[n, :, :, :, :, :] = variance[n, :, :, :, :, :] * std
    
    if freq:
        log_var = torch.log(variance)
        return log_var
    else:
        return variance


# display all error maps and image sets
# error (the quantity to be plotted) has to be normalized already
def display_map(error, title, filename, plot_min, plot_max):
    num = 0
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    for col in range(2):
        for row in range(2):
            im = ax[col, row].imshow(error[num, 0, 0, 0, :, :].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
            ax[col, row].set_xticks([])
            ax[col, row].set_yticks([])
            num += 1
            ax[col, row].set_title(f"{num}", y=0.95, fontsize=8)
    # colorbar
    fig.subplots_adjust(left=0.0,
                            bottom=0.05, 
                            right=0.3, 
                            top=0.9, 
                            wspace=0.0, 
                            hspace=0.2)
    cbar_ax = fig.add_axes([0.31, 0.05, 0.01, 0.8])
    color_bar = fig.colorbar(im, cax=cbar_ax)
    color_bar.minorticks_on()

    fig.suptitle(title, x=0.15, fontsize=10)
    plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.2)
    print(filename + " saved")

    return 0

# average the error maps over pixels (dimensions 4 and 5), get one error for each patient
def avg_across_pixels(error_maps):
    errors = torch.mean(error_maps, dim=(4,5))
    return errors

# average and std among the patients
def avg_across_patients(errors):
    error = torch.mean(errors, dim=0)
    std = torch.std(errors, dim=0)
    return error, std