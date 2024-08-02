import torch
# import matplotlib.pyplot as plt
# from matplotlib import colors
# from common_large import get_diffusion_bridge_model, load_weights
# import laboratory_tcga as lab
from scipy.ndimage import gaussian_filter
from map_fn_new import *

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

# BANDPASS
def bandpass(folder, nums, image_sets, recon_filtered_all):
    print("band pass")
    for u in range(1, 13):
        plot_min = -10
        plot_max = 2

        true_images, measurements, reconstructions = image_sets

        band_filtered = recon_filtered_all[u - 1] - recon_filtered_all[u]
        recon_filtered = torch.from_numpy(band_filtered)
        image_sets = true_images, measurements, recon_filtered

        # mean of the filtered reconstruction
        mean = calculate_mean(nums, image_sets)
        variance = calculate_variance(mean, nums, image_sets, freq=True)
        title = f"Variance maps (band filtered: sigma={round(0.4*(u-1), 1)}-{round(0.4*u, 1)})"
        file = f"var_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}.png"

        # print("max: ", torch.max(variance))
        # print("min: ", torch.min(variance))

        # print(f"var_{round(0.4*(u-1), 1)}-{round(0.4*u, 1)}")
        # hist_var, edge_var = torch.histogram(variance[0, 0, 0, :, :, :], bins=100)
        # print(hist_var, edge_var)

        display_map(variance, title, folder + file, plot_min, plot_max)

    return 0


# LOWPASS
def lowpass(folder, nums, image_sets, recon_filtered_all):
    print("low pass")
    for u in range(13):
        plot_min = -10
        plot_max = 2

        true_images, measurements, reconstructions = image_sets

        lowpass_filtered = recon_filtered_all[u]
        recon_filtered = torch.from_numpy(lowpass_filtered)
        image_sets = true_images, measurements, recon_filtered

        mean = calculate_mean(nums, image_sets)
        variance = calculate_variance(mean, nums, image_sets, freq=True)
        title = f"Variance maps (lowpass filtered: sigma={round(0.4*u, 1)})"
        file = f"var_{round(0.4*u, 1)}.png"
        display_map(variance, title, folder + file, plot_min, plot_max)

        # print("max: ", torch.max(variance))
        # print("min: ", torch.min(variance))

    return 0



