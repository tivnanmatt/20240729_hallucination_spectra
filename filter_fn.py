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
    # FWHM=2mm
    LPF_1 = gaussian_filter(reconstructions, sigma=0.84925690021)
    # FWHM=8mm
    LPF_2 = gaussian_filter(reconstructions, sigma=3.39702760085)

    BPF_1 = LPF_2
    BPF_2 = LPF_1 - LPF_2
    BPF_3 = 1 - LPF_1
    band_filtered = [BPF_1, BPF_2, BPF_3]
    lowpass_filtered = [LPF_1, LPF_2]

    return band_filtered, lowpass_filtered

# def bandpass(folder, nums, image_sets, recon_filtered_all):

# BANDPASS
def bandpass(folder, nums, image_sets, band_filtered):
    print("band pass")
    bands = ["low", "middle", "high"]
    for u in range(len(band_filtered)):
        
        true_images, measurements, reconstructions = image_sets

        band_filtered_img = band_filtered[u]
        recon_filtered = torch.from_numpy(band_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        # mean of the filtered reconstruction
        mean = calculate_mean(nums, image_sets)
        variance = calculate_variance(mean, nums, image_sets, freq=False)
        title = "STD maps (band filtered: " + bands[u] + ")"
        file = "std_" + bands[u] + ".png"
        # title = f"STD maps (band filtered: sigma={round(0.4*(u-1), 1)}-{round(0.4*u, 1)})"
        # file = f"std_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}.png"

        # use the max and min pixel values as the color bar limits
        plot_min = torch.round(torch.min(variance))
        plot_max = torch.round(torch.max(variance))

        display_map(variance, title, folder + file, plot_min, plot_max)

    return 0


# LOWPASS
def lowpass(folder, nums, image_sets, lowpass_filtered):
    print("low pass")
    FWHM = [2, 8]
    for u in range(len(lowpass_filtered)):

        true_images, measurements, reconstructions = image_sets

        lowpass_filtered_img = lowpass_filtered[u]
        recon_filtered = torch.from_numpy(lowpass_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        mean = calculate_mean(nums, image_sets)
        variance = calculate_variance(mean, nums, image_sets, freq=False)
        
        title = f"STD maps (lowpass filtered: sigma={FWHM[u]})"
        file = f"std_{FWHM[u]}.png"
        plot_min = torch.round(torch.min(variance))
        plot_max = torch.round(torch.max(variance))
        display_map(variance, title, folder + file, plot_min, plot_max)

    return 0

"""
check histograms
print("max: ", torch.max(variance))
print("min: ", torch.min(variance))

print(f"std_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}")
hist_var, edge_var = torch.histogram(variance[0, 0, 0, :, :, :], bins=100)
print(hist_var, edge_var)
print(variance[0, 0, 0, :, :, :])
"""

# BANDPASS
def bandpass_roi(folder, nums, image_sets, band_filtered, rois):
    print("band pass")
    bands = ["low", "middle", "high"]
    for u in range(len(band_filtered)):
        
        true_images, measurements, reconstructions = image_sets

        band_filtered_img = band_filtered[u]
        recon_filtered = torch.from_numpy(band_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        # mean of the filtered reconstruction
        mean = calculate_roi_mean(nums, image_sets, rois)
        variance = calculate_roi_variance(mean, nums, image_sets, rois, freq=False)
        title = "STD maps (band filtered: " + bands[u] + ")"
        file = "std_" + bands[u] + "_roi.png"
        # title = f"STD maps (band filtered: sigma={round(0.4*(u-1), 1)}-{round(0.4*u, 1)})"
        # file = f"std_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}.png"

        # use the max and min pixel values as the color bar limits
        plot_min = torch.round(torch.min(variance))
        plot_max = torch.round(torch.max(variance))

        display_map_roi(variance, title, folder + file, plot_min, plot_max, rois, crop=False)

    return 0


# LOWPASS
def lowpass_roi(folder, nums, image_sets, lowpass_filtered, rois):
    print("low pass")
    FWHM = [2, 8]
    for u in range(len(lowpass_filtered)):

        true_images, measurements, reconstructions = image_sets

        lowpass_filtered_img = lowpass_filtered[u]
        recon_filtered = torch.from_numpy(lowpass_filtered_img)
        image_sets = true_images, measurements, recon_filtered

        mean = calculate_roi_mean(nums, image_sets, rois)
        variance = calculate_roi_variance(mean, nums, image_sets, rois, freq=False)
        
        title = f"STD maps (lowpass filtered: FWHM={FWHM[u]})"
        file = f"std_{FWHM[u]}_roi.png"
        plot_min = torch.round(torch.min(variance))
        plot_max = torch.round(torch.max(variance))
        
        display_map_roi(variance, title, folder + file, plot_min, plot_max, rois, crop=False)

    return 0

