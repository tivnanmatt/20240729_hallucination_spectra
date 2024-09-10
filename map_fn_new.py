"""
map_fn_new.py: Contains functions to calculate and display the RMSE, bias, and STD maps of 
reconstructions. All maps are displayed in grayscale. 
"""
import torch
import matplotlib.pyplot as plt
import laboratory as lab


# get the mean and std of all images from the data loader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = lab.torch.datasets.TCGA(
                            root='TCGA_LIHC',
                            train=False).to(device)
mu = loader.mu
print("loader mean:", mu)
sig = loader.sigma
print("loader std:", sig)

digit_pixels = 28
plot_min = -160
plot_max = 240


def get_mean_std():
    """
    Get the average and standard deviation of the input images
    """
    return mu, sig


def calculate_mean(nums, image_sets):
    """
    Find the mean across reconstructions
    """
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    mean = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels, dtype=torch.complex64)
    # for every sample image
    for n in range(num_images):
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                mean[n, 0, 0, :, :, :] += reconstructions[n, m, r, :, :, :].detach().cpu()
        # mean of this sample image
        mean[n, :, :, :, :, :] /= (num_measurements * num_reconstructions)

    return mean


def calculate_rmse(nums, image_sets, freq=False):
    """
    Find the RMSE between the true images and the reconstructions
    """
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    rmse = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels)
    for n in range(num_images):
        total_diff = torch.zeros(1, 1, 1, 1, num_pixels, num_pixels)
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                # find the difference between each reconstruction and the ground truth
                diff_squared = (true_images[n, :, :, :, :, :].detach().cpu() - reconstructions[n, m, r, :, :, :].detach().cpu()).abs() ** 2
                total_diff += diff_squared

        # MSE for this ground truth image
        rmse[n, :, :, :, :, :] = total_diff / (num_measurements * num_reconstructions)
        # RMSE
        rmse[n, :, :, :, :, :] = torch.sqrt(rmse[n, :, :, :, :, :])
        # reverse the normalization to HU
        rmse[n, :, :, :, :, :] = rmse[n, :, :, :, :, :] * sig
    
    if freq:
        log_rmse = torch.log(rmse)
        return log_rmse
    else:
        return rmse


def calculate_bias(mean, nums, image_sets, freq=False):
    """
    Find the bias of the sampling by comparing the mean of the reconstructions and the true images
    """
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    bias = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels)
    for m in range(num_images):
        bias[m, :, :, :, :, :] = (mean[m, :, :, :, :, :] - true_images[m, :, :, :, :, :].detach().cpu()).abs()
        # reverse normalization
        bias[m, :, :, :, :, :] = bias[m, :, :, :, :, :] * sig

    if freq:
        log_bias = torch.log(bias)
        return log_bias
    else:
        return bias


def calculate_std(mean, nums, image_sets, freq=False):
    """
    Find the STD of the sampling by comparing the reconstructions with their mean
    """
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    std = torch.zeros(num_images, 1, 1, 1, num_pixels, num_pixels)
    for n in range(num_images):
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                std[n, :, :, :, :, :] += (reconstructions[n, m, r, :, :, :].detach().cpu() - mean[n, :, :, :, :, :]).abs() ** 2
        # variance among the reconstructions of this sample image
        if (num_measurements * num_reconstructions - 1) == 0:
            divisor = 1
        else:
            divisor = (num_measurements * num_reconstructions - 1)
        std[n, :, :, :, :, :] = torch.div(std[n, :, :, :, :, :], divisor) 
        # use std
        std[n, :, :, :, :, :] = torch.sqrt(std[n, :, :, :, :, :])
        # reverse the normalization 
        std[n, :, :, :, :, :] = std[n, :, :, :, :, :] * sig
    
    if freq:
        log_std = torch.log(std)
        return log_std
    else:
        return std


def display_map(error, title, filename, plot_min, plot_max):
    """
    Display all error maps and image sets
    Error (the quantity to be plotted) has to be normalized already
    """
    num = 0
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))
    for col in range(4):
        for row in range(4):
            
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


def display_single(error, title, filename, plot_min, plot_max):
    """
    Display a single image
    """
    num = 0
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    im = plt.imshow(error[num, 0, 0, 0, :, :].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    num+=1
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


def display_recon(error, title, filename, plot_min, plot_max, rois):
    """
    Display four reconstructions from the first measurement of the first patient
    """
    num = 0
    iRow, iCol = rois[0]
    fig, ax = plt.subplots(2, 2, figsize=(15, 5))
    for col in range(2):
        for row in range(2):
            iRow, iCol = rois[0]
            im = ax[col, row].imshow(error[0, 0, num, 0, :, :].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
            # uncomment the below line to plot the ROIs
            # im = ax[col, row].imshow(error[0, 0, num, 0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
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


def avg_across_pixels(error_maps):
    """
    Average the error maps over pixels (dimensions 4 and 5), get one error for each patient
    """
    errors = torch.mean(error_maps, dim=(4,5))
    return errors


def avg_across_patients(errors):
    """
    Average and std among the patients
    """
    error = torch.mean(errors, dim=0)
    std = torch.std(errors, dim=0)
    return error, std