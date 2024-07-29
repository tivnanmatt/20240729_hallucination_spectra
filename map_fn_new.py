"""
map_functions.py: Contains functions to calculate and display the MSE, bias-squared, and variance maps of 
reconstructions. All maps are displayed in grayscale. 
"""
import torch
import matplotlib.pyplot as plt
import laboratory_tcga as lab

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loader = lab.torch.datasets.TCGA(
                            root='training_large',
                            train=False).to(device)
mean = loader.mu
print("loader mean:", mean)
std = loader.sigma
print("loader std:", std)


# MSE
def calculate_mse(num_images, num_reconstructions, num_pixel, true_images, reconstructions, freq=False):
    mse = torch.zeros(num_images, 1, 1, 1, num_pixel, num_pixel)
    for m in range(num_images):
        total_diff = torch.zeros(1, 1, 1, 1, num_pixel, num_pixel)
        for n in range(num_reconstructions):
            # find the difference between each reconstruction and the ground truth
            # diff_squared, total_diff, mse are all tensors of the same shape as the images
            diff_squared = (true_images[m, :, :, :, :, :] - reconstructions[m, :, n, :, :, :]).abs() ** 2
            total_diff += diff_squared
        # MSE for this ground truth image
        mse[m, :, :, :, :, :] = total_diff / num_reconstructions
        # reverse the normalization to HU
        mse[m, :, :, :, :, :] = mse[m, :, :, :, :, :] * (std ** 2)

    return mse


# Display MSE map
def display_mse(mse, cmin, cmax, freq=True):
    num = 0
    print("min", torch.min(mse[11, 0, 0, 0, :, :]))
    print("max", torch.max(mse[15, 0, 0, 0, :, :]))
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))
    for col in range(4):
        for row in range(4):
            # im = ax[col, row].imshow(mse[num, 0, 0, 0, :, :], cmap='gray', vmin=cmin, vmax=cmax)
            im = ax[col, row].imshow(mse[num, 0, 0, 0, :, :], cmap='gray')
            # remove ticks
            ax[col, row].set_xticks([])
            ax[col, row].set_yticks([])
            # add title
            num += 1
            ax[col, row].set_title(f"{num}", y=0.95, fontsize=8)
    # colorbar
    fig.subplots_adjust(left=0.0,
                            bottom=0.05, 
                            right=0.3, 
                            top=0.9, 
                            wspace=0.0, 
                            hspace=0.2)
    cbar_ax = fig.add_axes([0.31, 0.05, 0.01, 0.2])
    color_bar = fig.colorbar(im, cax=cbar_ax)
    color_bar.minorticks_on()
    if freq:
        fig.suptitle("MSE maps (large) (frequency)", x=0.15, fontsize=10)
        plt.savefig("mse_freq_large.png")
        print("mse_freq_large.png saved")
    else:
        fig.suptitle("MSE maps (large)", x=0.15, fontsize=10)
        plt.savefig("mse_large_test_scale.png")
        print("mse_large.png saved")

    return 0


# Find the mean across reconstructions
def calculate_mean(num_images, num_reconstructions, num_pixel, reconstructions):
    mean = torch.zeros(num_images, 1, 1, 1, num_pixel, num_pixel, dtype=torch.complex64)
    # for every sample image
    for m in range(num_images):
        total = torch.zeros(1, 1, 1, 1, num_pixel, num_pixel, dtype=torch.complex64)
        # for every reconstruction
        for n in range(num_reconstructions):
            # find the mean across all reconstructions
            total += reconstructions[m, :, n, :, :, :]
        # mean of this sample image
        mean[m, :, :, :, :, :] = total / num_reconstructions

    return mean


# Bias-squared 
def calculate_bias(num_images, num_pixel, mean, true_images):
    bias = torch.zeros(num_images, 1, 1, 1, num_pixel, num_pixel)
    print(bias[0, :, :, :, :, :])
    for m in range(num_images):
        bias[m, :, :, :, :, :] = (mean[m, :, :, :, :, :] - true_images[m, :, :, :, :, :]).abs() ** 2
        # reverse normalization
        bias[m, :, :, :, :, :] = bias[m, :, :, :, :, :] * (std ** 2)

    return bias


# Display bias-squared maps
def display_bias(log_bias, bmin, bmax, freq=True):
    num = 0
    print("min", torch.min(log_bias[11, 0, 0, 0, :, :]))
    print("max", torch.max(log_bias[15, 0, 0, 0, :, :]))
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))
    for col in range(4):
        for row in range(4):
            # im = ax[col, row].imshow(log_bias[num, 0, 0, 0, :, :], cmap='gray', vmin=bmin, vmax=bmax)
            im = ax[col, row].imshow(log_bias[num, 0, 0, 0, :, :], cmap='gray')
            # remove ticks
            ax[col, row].set_xticks([])
            ax[col, row].set_yticks([])
            # add title
            num += 1
            ax[col, row].set_title(f"{num}", y=0.95, fontsize=8)
    # colorbar
    fig.subplots_adjust(left=0.0,
                            bottom=0.05, 
                            right=0.3, 
                            top=0.9, 
                            wspace=0.0, 
                            hspace=0.2)
    cbar_ax = fig.add_axes([0.31, 0.05, 0.01, 0.2])
    color_bar = fig.colorbar(im, cax=cbar_ax)
    color_bar.minorticks_on()
    if freq:
        fig.suptitle("Bias-squared maps (large) (frequency)", x=0.15, fontsize=10)
        plt.savefig("bias_2_freq_large.png")
        print("bias_2_freq_large.png saved")
    else: 
        fig.suptitle("Bias-squared maps (large)", x=0.15, fontsize=10)
        plt.savefig("bias_2_large_test_scale.png")
        print("bias_2_large.png saved")

    return 0


# Variance
def calculate_variance(num_images, num_pixel, mean, num_reconstructions, reconstructions):
    variance = torch.zeros(num_images, 1, 1, 1, num_pixel, num_pixel)
    for m in range(num_images):
        total_var = torch.zeros(1, 1, 1, 1, num_pixel, num_pixel)
        # for each reconstruction
        for n in range(num_reconstructions):
            total_var += (reconstructions[m, :, n, :, :, :] - mean[m, :, :, :, :, :]).abs() ** 2
        # variance among the reconstructions of this sample image
        variance[m, :, :, :, :, :] = total_var / (num_reconstructions - 1)
        # reverse the normalization 
        variance[m, :, :, :, :, :] = variance[m, :, :, :, :, :] * (std ** 2)

    return variance


# Display variance map
def display_variance(log_variance, varmin, varmax, freq=True, bandpass=False, lowpass=False, u=0):
    num = 0
    print("min", torch.min(log_variance[11, 0, 0, 0, :, :]))
    print("max", torch.max(log_variance[15, 0, 0, 0, :, :]))
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))
    for col in range(4):
        for row in range(4):
            # im = ax[col, row].imshow(log_variance[num, 0, 0, 0, :, :], cmap='gray', vmin=varmin, vmax=varmax)
            im = ax[col, row].imshow(log_variance[num, 0, 0, 0, :, :], cmap='gray')
            # remove ticks
            ax[col, row].set_xticks([])
            ax[col, row].set_yticks([])
            # add title
            num += 1
            ax[col, row].set_title(f"{num}", y=0.95, fontsize=8)
    # colorbar
    fig.subplots_adjust(left=0.0,
                            bottom=0.05, 
                            right=0.3, 
                            top=0.9, 
                            wspace=0.0, 
                            hspace=0.2)
    cbar_ax = fig.add_axes([0.31, 0.05, 0.01, 0.2])
    color_bar = fig.colorbar(im, cax=cbar_ax)
    color_bar.minorticks_on()
    
    if freq:
        fig.suptitle("Variance maps (large) (frequency)", x=0.15, fontsize=10)
        plt.savefig("variance_freq_large.png")
        print("variance_freq_large.png saved")

    if bandpass:
        title = f"Variance maps (bandpass filtered: sigma {round(0.4*(u-1), 1)}-{round(0.4*u, 1)})"
        fig.suptitle(title, x=0.15, fontsize=10)
        filename = f"var_filtered_band_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}.png"
        plt.savefig(filename)
        print(f"{round(0.4*(u-1), 1)}-{round(0.4*u, 1)} saved")

    if lowpass:
        title = f"Variance maps (lowpass filtered: sigma={round(0.4*u, 1)})"
        fig.suptitle(title, x=0.15, fontsize=10)
        filename = f"var_filtered_low_{round(0.4*u, 1)}.png"
        plt.savefig(filename)
        print(f"{round(0.4*u, 1)} saved")

    else:
        fig.suptitle("Variance maps (large)", x=0.15, fontsize=10)
        plt.savefig("variance_large.png")
        print("variance_large.png saved")

    return 0

def display_variance_measurement(log_variance, varmin, varmax, freq=True, bandpass=False, lowpass=False, u=0):
    num = 0
    print("min", torch.min(log_variance[0, 0, 0, 0, :, :]))
    print("max", torch.max(log_variance[0, 0, 0, 0, :, :]))
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))
    for col in range(4):
        for row in range(4):
            # im = ax[col, row].imshow(log_variance[num, 0, 0, 0, :, :], cmap='gray', vmin=varmin, vmax=varmax)
            im = ax[col, row].imshow(log_variance[num, 0, 0, 0, :, :], cmap='gray')
            # remove ticks
            ax[col, row].set_xticks([])
            ax[col, row].set_yticks([])
            # add title
            num += 1
            ax[col, row].set_title(f"{num}", y=0.95, fontsize=8)
    # colorbar
    fig.subplots_adjust(left=0.0,
                            bottom=0.05, 
                            right=0.3, 
                            top=0.9, 
                            wspace=0.0, 
                            hspace=0.2)
    cbar_ax = fig.add_axes([0.31, 0.05, 0.01, 0.2])
    color_bar = fig.colorbar(im, cax=cbar_ax)
    color_bar.minorticks_on()
    
    if freq:
        fig.suptitle("Variance maps (large) (frequency)", x=0.15, fontsize=10)
        plt.savefig("variance_freq_large.png")
        print("variance_freq_large.png saved")

    if bandpass:
        title = f"Variance maps (bandpass filtered: sigma {round(0.4*(u-1), 1)}-{round(0.4*u, 1)})"
        fig.suptitle(title, x=0.15, fontsize=10)
        filename = f"var_filtered_band_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}.png"
        plt.savefig(filename)
        print(f"{round(0.4*(u-1), 1)}-{round(0.4*u, 1)} saved")

    if lowpass:
        title = f"Variance maps (lowpass filtered: sigma={round(0.4*u, 1)})"
        fig.suptitle(title, x=0.15, fontsize=10)
        filename = f"var_filtered_low_{round(0.4*u, 1)}.png"
        plt.savefig(filename)
        print(f"{round(0.4*u, 1)} saved")

    else:
        fig.suptitle("Variance maps (measurements) (large)", x=0.15, fontsize=10)
        plt.savefig("variance_measurements.png")
        print("saved variance_measurements.png")

    return 0


# Display true images
def display_true(true_images, mode):
    print("min", torch.min(true_images[11, 0, 0, 0, :, :]))
    print("max", torch.max(true_images[15, 0, 0, 0, :, :]))
    num = 0
    fig, ax = plt.subplots(4, 4, figsize=(15, 5))
    true_images = true_images * std  + mean
    for col in range(4):
        for row in range(4):
            # im = ax[col, row].imshow(true_images[num, 0, 0, 0, :, :], cmap='gray', vmin=-160, vmax=240)
            im = ax[col, row].imshow(true_images[num, 0, 0, 0, :, :], cmap='gray', vmin=-160, vmax=240)
            # remove ticks
            ax[col, row].set_xticks([])
            ax[col, row].set_yticks([])
            # add title
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


    if mode == 0:
        title = "True images"
        fig.suptitle(title, x=0.15, fontsize=10)
        # plt.tight_layout()
        filename = "true.png"
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.2)
        print(filename, " saved")
    if mode == 1:
        title = "Measurements"
        fig.suptitle(title, x=0.15, fontsize=10)
        filename = "measurements.png"
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.2)
        print(filename, " saved")
    if mode == 2:
        title = "Reconstructions"
        fig.suptitle(title, x=0.15, fontsize=10)
        filename = "reconstuctions.png"
        plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.2)
        print(filename, " saved")

