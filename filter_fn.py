import torch
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from common_large import get_diffusion_bridge_model, load_weights
# import laboratory_tcga as lab
from scipy.ndimage import gaussian_filter
from map_fn_new import *
from map_fn_roi import *

# add to eval_fn, call in evaluation?
# apply frequency filter to reconstructions (non-fft!)
def filter_recon(image_sets):
    true_images, measurements, reconstructions = image_sets
    reconstructions = reconstructions.detach().cpu()
    recon_filtered_1 = gaussian_filter(reconstructions, sigma=0.0)
    recon_filtered_2 = gaussian_filter(reconstructions, sigma=0.4)
    recon_filtered_3 = gaussian_filter(reconstructions, sigma=0.8)
    recon_filtered_4 = gaussian_filter(reconstructions, sigma=1.2)
    recon_filtered_5 = gaussian_filter(reconstructions, sigma=1.6)
    recon_filtered_6 = gaussian_filter(reconstructions, sigma=2.0)
    recon_filtered_7 = gaussian_filter(reconstructions, sigma=2.4)
    recon_filtered_8 = gaussian_filter(reconstructions, sigma=2.8)
    recon_filtered_9 = gaussian_filter(reconstructions, sigma=3.2)
    recon_filtered_10 = gaussian_filter(reconstructions, sigma=3.6)
    recon_filtered_11 = gaussian_filter(reconstructions, sigma=4.0)
    recon_filtered_12 = gaussian_filter(reconstructions, sigma=4.4)
    recon_filtered_13 = gaussian_filter(reconstructions, sigma=4.8)
    recon_filtered_all = [recon_filtered_1, recon_filtered_2, recon_filtered_3, recon_filtered_4, recon_filtered_5, recon_filtered_6, recon_filtered_7, recon_filtered_8, 
                      recon_filtered_9, recon_filtered_10, recon_filtered_11, recon_filtered_12, recon_filtered_13]
    return recon_filtered_all

def three_filters(image_sets):
    true_images, measurements, reconstructions = image_sets
    reconstructions = reconstructions.detach().cpu()
    # FWHM=1.1mm
    LPF_1 = gaussian_filter(reconstructions, sigma=0.46709129511)
    # FWHM=1.5mm
    LPF_2 = gaussian_filter(reconstructions, sigma=0.63694267515)

    BPF_1 = LPF_2
    BPF_2 = LPF_1 - LPF_2
    BPF_3 = reconstructions.numpy() - LPF_1
    band_filtered = [BPF_1, BPF_2, BPF_3]
    lowpass_filtered = [LPF_1, LPF_2]

    return band_filtered, lowpass_filtered


def filter_maps(folder, nums, image_sets):
    band_filtered, lowpass_filtered = three_filters(image_sets)
    band_rmses, band_biases, band_stds = bandpass(folder, nums, image_sets, band_filtered)
    low_std = lowpass(folder, nums, image_sets, lowpass_filtered)

    return band_rmses, band_biases, band_stds, low_std

# def bandpass(folder, nums, image_sets, recon_filtered_all):

# BANDPASS
def bandpass(folder, nums, image_sets, band_filtered):
    print("band pass")
    bands = ["low", "middle", "high"]
    band_rmses = []
    band_biases = []
    band_stds = []
    for u in range(len(band_filtered)):
        
        true_images, measurements, reconstructions = image_sets

        band_filtered_img = band_filtered[u]
        recon_filtered = torch.from_numpy(band_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        # mean of the filtered reconstruction
        mean = calculate_mean(nums, image_sets)
        rmse = calculate_rmse(nums, image_sets, freq=False)
        bias = calculate_bias(mean, nums, image_sets, freq=False)
        std = calculate_std(mean, nums, image_sets, freq=False)
        title = "STD maps (band filtered: " + bands[u] + ")"
        file = "std_" + bands[u] + ".png"
        # title = f"STD maps (band filtered: sigma={round(0.4*(u-1), 1)}-{round(0.4*u, 1)})"
        # file = f"std_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}.png"

        # use the max and min pixel values as the color bar limits
        plot_min = torch.round(torch.min(std))
        plot_max = torch.round(torch.max(std))

        display_map(std, title, folder + file, plot_min, plot_max)

        rmse_vector, rmse_avg, rmse_std = error_avg(rmse)
        band_rmse = rmse_vector, rmse_avg, rmse_std
        band_rmses.append(band_rmse)

        bias_vector, bias_avg, bias_std = error_avg(bias)
        band_bias = bias_vector, bias_avg, bias_std
        band_biases.append(band_bias)

        # calculate std vectors, final average std, std of std
        std_vector, std_avg, std_std = error_avg(std) 
        band_std = std_vector, std_avg, std_std
        band_stds.append(band_std)
        # need a list of band_std for the three bands, now only have one band; same for lowpass
    return band_rmses, band_biases, band_stds


# LOWPASS
def lowpass(folder, nums, image_sets, lowpass_filtered):
    print("low pass")
    FWHM = [1.1, 1.5]
    for u in range(len(lowpass_filtered)):

        true_images, measurements, reconstructions = image_sets

        lowpass_filtered_img = lowpass_filtered[u]
        recon_filtered = torch.from_numpy(lowpass_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        mean = calculate_mean(nums, image_sets)
        std = calculate_std(mean, nums, image_sets, freq=False)
        
        title = f"STD maps (lowpass filtered: FWHM={FWHM[u]})"
        file = f"std_{FWHM[u]}.png"
        plot_min = torch.round(torch.min(std))
        plot_max = torch.round(torch.max(std))
        display_map(std, title, folder + file, plot_min, plot_max)
        # calculate std vectors, final average std, std of std
        std_vector, std_avg, std_std = error_avg(std) 
        low_std = std_vector, std_avg, std_std

    return low_std

"""
check histograms
print("max: ", torch.max(variance))
print("min: ", torch.min(variance))

print(f"std_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}")
hist_var, edge_var = torch.histogram(variance[0, 0, 0, :, :, :], bins=100)
print(hist_var, edge_var)
print(variance[0, 0, 0, :, :, :])
"""

def filter_maps_roi(folder, nums, image_sets, rois):
    band_filtered, lowpass_filtered = three_filters(image_sets)
    rmses, rmse_stds, biases, bias_stds, stds, std_stds = bandpass_roi(folder, nums, image_sets, band_filtered, rois)
    low_std = lowpass_roi(folder, nums, image_sets, lowpass_filtered, rois)

    return rmses, rmse_stds, biases, bias_stds, stds, std_stds

# BANDPASS
def bandpass_roi(folder, nums, image_sets, band_filtered, rois):
    print("band pass")
    bands = ["low", "middle", "high"]
    band_rmses = []
    band_biases = []
    band_stds = []

    rmses = []
    rmse_stds = []
    biases = []
    bias_stds = []
    stds = []
    std_stds = []
    for u in range(len(band_filtered)):
        
        true_images, measurements, reconstructions = image_sets

        band_filtered_img = band_filtered[u]
        recon_filtered = torch.from_numpy(band_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        # mean of the filtered reconstruction
        mean = calculate_roi_mean(nums, image_sets, rois)
        rmse = calculate_roi_rmse(nums, image_sets, rois, freq=False)
        bias = calculate_roi_bias(mean, nums, image_sets, rois, freq=False)
        std = calculate_roi_std(mean, nums, image_sets, rois, freq=False)
        title = "STD maps (band filtered: " + bands[u] + ")"
        file = "std_" + bands[u] + "_roi.png"
        # title = f"STD maps (band filtered: sigma={round(0.4*(u-1), 1)}-{round(0.4*u, 1)})"
        # file = f"std_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}.png"

        # use the max and min pixel values as the color bar limits
        # plot_min = torch.round(torch.min(std))
        # plot_max = torch.round(torch.max(std))

        plot_min = 0
        plot_max = 40

        display_map_roi(std, title, folder + file, plot_min, plot_max, rois, crop=False)

        rmse_vector, rmse_avg, rmse_std = error_avg(rmse)
        band_rmse = rmse_vector, rmse_avg, rmse_std
        band_rmses.append(band_rmse)

        rmses.append(rmse_avg)
        rmse_stds.append(rmse_std)

        bias_vector, bias_avg, bias_std = error_avg(bias)
        band_bias = bias_vector, bias_avg, bias_std
        band_biases.append(band_bias)

        biases.append(bias_avg)
        bias_stds.append(bias_std)

        # calculate std vectors, final average std, std of std
        std_vector, std_avg, std_std = error_avg(std) 
        band_std = std_vector, std_avg, std_std
        band_stds.append(band_std)

        stds.append(std_avg)
        std_stds.append(std_std)

    # return band_rmses, band_biases, band_stds
    return rmses, rmse_stds, biases, bias_stds, stds, std_stds   


# LOWPASS
def lowpass_roi(folder, nums, image_sets, lowpass_filtered, rois):
    print("low pass")
    FWHM = [1.1, 1.5]
    for u in range(len(lowpass_filtered)):

        true_images, measurements, reconstructions = image_sets

        lowpass_filtered_img = lowpass_filtered[u]
        recon_filtered = torch.from_numpy(lowpass_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        mean = calculate_roi_mean(nums, image_sets, rois)
        std = calculate_roi_std(mean, nums, image_sets, rois, freq=False)
        
        title = f"STD maps (lowpass filtered: FWHM={FWHM[u]})"
        file = f"std_{FWHM[u]}_roi.png"
        plot_min = torch.round(torch.min(std))
        plot_max = torch.round(torch.max(std))
        
        display_map_roi(std, title, folder + file, plot_min, plot_max, rois, crop=False)

        # calculate std vectors, final average std, std of std
        std_vector, std_avg, std_std = error_avg(std) 
        low_std = std_vector, std_avg, std_std

    return low_std

def error_avg(error_maps):
    # average across pixels
    errors = avg_across_pixels(error_maps)
    # average across patients
    error, std = avg_across_patients(errors)

    return errors, error, std