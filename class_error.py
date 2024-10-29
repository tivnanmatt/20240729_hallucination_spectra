"""
calculate the RMSE, bias, and STD for the 100 classifier images
"""

import torch
import numpy as np
from map_fn_roi import *
from map_fn_new import avg_across_pixels
from filter_fn import *

image_sets = torch.load("classifier_100/image_sets_a_d.pt")
rois = torch.load("classifier_100/rois.pt")
# labels = torch.load("classifier_100/ground_truth_label.pt")
true_images, measurements, reconstructions = image_sets
folder = "classifier_100/"
nums = 100, 1, 16, 512, 32

mean = calculate_roi_mean(nums, image_sets, rois)
rmse = calculate_roi_rmse(nums, image_sets, rois, freq=False)
bias = calculate_roi_bias(mean, nums, image_sets, rois, freq=False)
std = calculate_roi_std(mean, nums, image_sets, rois, freq=False)
print("errors calculated")

torch.save(mean, folder + "mean.pt")
torch.save(rmse, folder + "rmse.pt")
torch.save(bias, folder + "bias.pt")
torch.save(std, folder + "std.pt")

rmse_avg = avg_across_pixels(rmse).numpy().reshape((100, 1))
bias_avg = avg_across_pixels(bias).numpy().reshape((100, 1))
std_avg = avg_across_pixels(std).numpy().reshape((100, 1))

headerlist = "rmse, bias, std"
errors = np.concatenate((rmse_avg, bias_avg, std_avg), axis=1)
np.savetxt(folder + "errors.csv", errors, delimiter=",", header=headerlist)
# errors.to_csv(folder + "errors.csv", header=headerlist, sep=',')
# errors.tofile(folder + "errors.csv", sep=',')
print("errors.csv saved")

# filtered
band_rmses, band_biases, band_stds = filter_maps_roi(folder, nums, image_sets, rois)
print("filtered calculated")

torch.save(band_rmses, folder + "band_rmses.pt")
torch.save(band_biases, folder + "band_biases.pt")
torch.save(band_stds, folder + "band_stds.pt")

rmse_low, rmse_middle, rmse_high = band_rmses
bias_low, bias_middle, bias_high = band_biases
std_low, std_middle, std_high = band_stds

rmse_low_avg = avg_across_pixels(rmse_low).numpy().reshape((100, 1))
rmse_middle_avg = avg_across_pixels(rmse_middle).numpy().reshape((100, 1))
rmse_high_avg = avg_across_pixels(rmse_high).numpy().reshape((100, 1))

bias_low_avg = avg_across_pixels(bias_low).numpy().reshape((100, 1))
bias_middle_avg = avg_across_pixels(bias_middle).numpy().reshape((100, 1))
bias_high_avg = avg_across_pixels(bias_high).numpy().reshape((100, 1))

std_low_avg = avg_across_pixels(std_low).numpy().reshape((100, 1))
std_middle_avg = avg_across_pixels(std_middle).numpy().reshape((100, 1))
std_high_avg = avg_across_pixels(std_high).numpy().reshape((100, 1))

headerlistfiltered = "rmse_low, rmse_middle, rmse_high, bias_low, bias_middle, bias_high, std_low, std_middle, std_high"
errorsfiltered = np.concatenate((rmse_low_avg, rmse_middle_avg, rmse_high_avg,
                                 bias_low_avg, bias_middle_avg, bias_high_avg,
                                 std_low_avg, std_middle_avg, std_high_avg), axis=1)
np.savetxt(folder + "errors_filtered.csv", errorsfiltered, delimiter=",", header=headerlistfiltered)
# errorsfiltered.to_csv(folder + "errors_filtered.csv", header=headerlistfiltered, sep=',')
# errorsfiltered.tofile(folder + "errors_filtered.csv", sep=',')
print("errors_filtered.csv saved")

