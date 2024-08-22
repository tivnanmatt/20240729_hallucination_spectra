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

digit_pixels = 28

def get_mean_std():
    return mean, std

# Find the mean across reconstructions
def calculate_roi_mean(nums, image_sets, rois):
    # mean = torch.mean(reconstructions, 2, keepdim=True)
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    mean = torch.zeros(num_images, 1, 1, 1, digit_pixels, digit_pixels, dtype=torch.complex64)
    # for every sample image
    for n in range(num_images):
        iRow, iCol = rois[n]
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                # change the pixels
                mean[n, :, :, :, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)] += reconstructions[n, m, r, :, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu()
        # mean of this sample image
        mean[n, :, :, :, :, :] /= (num_measurements * num_reconstructions)

    return mean

# MSE
def calculate_roi_mse(nums, image_sets, rois, freq=False):
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    mse = torch.zeros(num_images, 1, 1, 1, digit_pixels, digit_pixels)
    for n in range(num_images):
        total_diff = torch.zeros(1, 1, 1, 1, digit_pixels, digit_pixels)
        iRow, iCol = rois[n]
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                # find the difference between each reconstruction and the ground truth
                # diff_squared, total_diff, mse are all tensors of the same shape as the images
                diff_squared = (true_images[n, :, :, :, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu() - reconstructions[n, m, r, :, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu()).abs() ** 2
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
def calculate_roi_bias(mean, nums, image_sets, rois, freq=False):
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    bias = torch.zeros(num_images, 1, 1, 1, digit_pixels, digit_pixels)
    for m in range(num_images):
        iRow, iCol = rois[m]
        # bias[m, :, :, :, :, :] = (mean[m, :, :, :, :, :] - true_images[m, :, :, :, :, :].detach().cpu()).abs() ** 2
        bias[m, :, :, :, :, :] = (mean[m, :, :, :, :, :] - true_images[m, :, :, :, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu()).abs()
        # reverse normalization
        # bias[m, :, :, :, :, :] = bias[m, :, :, :, :, :] * (std ** 2)
        bias[m, :, :, :, :, :] = bias[m, :, :, :, :, :] * std

    if freq:
        log_bias = torch.log(bias)
        return log_bias
    else:
        return bias


# Variance
def calculate_roi_variance(mean, nums, image_sets, rois, freq=False):
    num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps = nums
    true_images, measurements, reconstructions = image_sets
    variance = torch.zeros(num_images, 1, 1, 1, digit_pixels, digit_pixels)
    for n in range(num_images):
        # total_var = torch.zeros(1, 1, 1, 1, num_pixels, num_pixels)
        iRow, iCol = rois[n]
        for m in range(num_measurements):
            for r in range(num_reconstructions):
                # total_var[:, :, :, :, :, :] += (reconstructions[n, m, r, :, :, :].detach().cpu() - mean[n, :, :, :, :, :]).abs() ** 2
                variance[n, :, :, :, :, :] += (reconstructions[n, m, r, :, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu() - mean[n, :, :, :, :, :]).abs() ** 2
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
    
    
# new calculate error, new error maps
# copy from map_fn_new.py and modify from there
def error_maps_roi(folder, nums, image_sets, rois):
    # mean over reconstructions
    mean_roi = calculate_roi_mean(nums, image_sets, rois)
    
    # RMSE
    rmse_roi = calculate_roi_mse(nums, image_sets, rois, freq=False)
    display_map(rmse_roi, "RMSE maps in perturbed regions", folder + "rmse_roi.png", 0, 40)

    # Bias
    bias_roi = calculate_roi_bias(mean, nums, image_sets, freq=False)
    display_map(bias_roi, "Bias maps in perturbed regions", folder + "bias_roi.png", 0, 40)

    # STD
    std_roi = calculate_roi_variance(mean, nums, image_sets, freq=False)
    display_map(std_roi, "STD maps in perturbed regions", folder + "std_roi.png", 0, 40)

    # average across pixels and patients

    
    # filtered
    # recon_filtered_all = filter_recon(image_sets)
    # bandpass(folder, nums, image_sets, recon_filtered_all)
    # lowpass(folder, nums, image_sets, recon_filtered_all)

    return rmse_roi, bias_roi, std_roi


def error_freq_roi(folder, nums, image_sets, rois):
    # unpack parameters
    true_images, measurements, reconstructions = image_sets

    # take 2D FFT for the frequency domain
    true_freq = torch.fft.fft2(true_images)
    recon_freq = torch.fft.fft2(reconstructions)
    image_sets = true_freq, measurements, recon_freq

    mean_roi = calculate_roi_mean(nums, image_sets, rois)

    # MSE
    rmse_roi = calculate_roi_mse(nums, image_sets, rois, freq=True)
    display_map(rmse_roi, "RMSE maps in perturbed regions (frequency)", folder + "rmse_roi_freq.png", 4, 10)

    # Bias-squared
    bias_roi = calculate_roi_bias(mean_roi, nums, image_sets, freq=True)
    display_map(bias_roi, "Bias maps in perturbed regions (frequency)", folder + "bias_roi_freq.png", 4, 10)

    # Variance
    std_roi = calculate_roi_variance(mean_roi, nums, image_sets, freq=True)
    display_map(std_roi, "STD maps in perturbed regions (frequency)", folder + "std_roi_freq.png", 4, 10)
   
    return rmse_roi, bias_roi, std_roi

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


# display all error maps and image sets
# error (the quantity to be plotted) has to be normalized already
def display_map(error, title, filename, plot_min, plot_max):
    num = 0
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))
    for col in range(4):
        for row in range():
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


def plot_error_roi(folder, all_errors, all_errors_digit, frequency=False):
    rmse_avg, rmse_std, bias_avg, bias_std, std_avg, std_std = all_errors
    rmse_avg_d, rmse_std_d, bias_avg_d, bias_std_d, std_avg_d, std_std_d = all_errors_digit
    if frequency:
        bar_plot_error_roi("RMSE (frequency) in perturbed region", folder + "rmse_bar_f_roi.png", rmse_avg, rmse_std, rmse_avg_d, rmse_std_d)
        bar_plot_error_roi("Bias (frequency) in perturbed region", folder + "bias_bar_f_roi.png", bias_avg, bias_std, bias_avg_d, bias_std_d)
        bar_plot_error_roi("STD (frequency) in perturbed region", folder + "std_bar_f_roi.png", std_avg, std_std, std_avg_d, std_std_d)

    else:
        bar_plot_error_roi("RMSE in perturbed region", folder + "rmse_bar_roi.png", rmse_avg, rmse_std, rmse_avg_d, rmse_std_d)
        bar_plot_error_roi("Bias in perturbed region", folder + "bias_bar_roi.png", bias_avg, bias_std, bias_avg_d, bias_std_d)
        bar_plot_error_roi("STD in perturbed region", folder + "std_bar_roi.png", std_avg, std_std, std_avg_d, std_std_d)

    return 0

def bar_plot_error_roi(title, filename, error_1, std_1, error_2, std_2):
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