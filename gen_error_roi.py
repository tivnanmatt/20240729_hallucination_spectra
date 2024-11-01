"""
Plot error maps that focus on the digit-inserted regions
For both original true images and perturbed true image reconstructions
"""

from eval_fn import *
from sample_fn import *
from map_fn_roi import *
from filter_fn import *
from map_fn_new import *

folder_a = "samples_a_32/"
folder_a_d = "samples_a_d_32/"
record_a = 'record_a_roi.txt'
nums_a = 16, 1, 16, 512, 32
image_sets_a = torch.load("samples_a_32/image_sets_a.pt")
image_sets_a_d = torch.load("samples_a_d_32/image_sets_a_d.pt")
rois = torch.load("samples_a_d_32/rois.pt")

true, meas, recon = image_sets_a
true_d, meas_d, recon_d = image_sets_a_d
image_sets_a_1 = true[:, :, :, :, :, :], meas[:, :, :, :, :, :], recon[:, :, :16, :, :, :]
image_sets_a_2 = true[:, :, :, :, :, :], meas[:, :, :, :, :, :], recon[:, :, 16:, :, :, :]
image_sets_a_d_1 = true_d[:, :, :, :, :, :], meas_d[:, :, :, :, :, :], recon_d[:, :, :16, :, :, :]
image_sets_a_d_2 = true_d[:, :, :, :, :, :], meas_d[:, :, :, :, :, :], recon_d[:, :, 16:, :, :, :]

display_rois(folder_a, image_sets_a, rois)

# rmse, bias, std = error_maps_roi(folder_a, nums_a, image_sets_a, rois)
# all_errors, error_vectors = calculate_error(rmse, bias, std, frequency=False)
# record_errors(all_errors, record_a, frequency=False, perturbation=False)
# torch.save(error_vectors, folder_a + 'error_vectors_roi.pt')
# torch.save(all_errors, folder_a + 'all_errors_roi.pt')

# rmses, rmse_stds, biases, bias_stds, stds, std_stds = filter_maps_roi(folder_a, nums_a, image_sets_a, rois)
# band_stats = band_rmses, band_biases, band_stds
# torch.save(band_stats, folder_a + 'band_stats.pt')
# bar_band('RMSE', folder_a + 'rmse_bands.png', rmses, rmse_stds)

# low_std_vector, low_std_avg, low_std_std = low_std
# print(f"band_std_roi: {band_std_avg}, {band_std_std}")
# print(f"low_std_roi: {low_std_avg}, {low_std_std}")

# rmse_f, bias_f, std_f = error_freq_roi(folder_a, nums_a, image_sets_a, rois)
# all_errors_f, error_vectors_f = calculate_error(rmse_f, bias_f, std_f, frequency=True)
# record_errors(all_errors_f, record_a, frequency=True, perturbation=False)
# torch.save(error_vectors_f, folder_a + 'error_vectors_f_roi.pt')
# torch.save(all_errors_f, folder_a + 'all_errors_f_roi.pt')

# display_rois(folder_a_d, image_sets_a_d, rois)

# with digits
# rmse_d, bias_d, std_d = error_maps_roi(folder_a_d, nums_a, image_sets_a_d, rois)
# all_errors_d, error_vectors_d = calculate_error(rmse_d, bias_d, std_d, frequency=False)
# record_errors(all_errors_d, record_a, frequency=False, perturbation=True)
# torch.save(error_vectors_d, folder_a_d + 'error_vectors_d_roi.pt')
# torch.save(all_errors_d, folder_a_d + 'all_errors_d_roi.pt')

# band_rmses_d, band_biases_d, band_stds_d, low_std_d = filter_maps_roi(folder_a_d, nums_a, image_sets_a_d, rois)

# first set

rmses, rmse_stds, biases, bias_stds, stds, std_stds = filter_maps_roi(folder_a, nums_a, image_sets_a_2, rois)
rmses_d, rmse_stds_d, biases_d, bias_stds_d, stds_d, std_stds_d = filter_maps_roi(folder_a_d, nums_a, image_sets_a_d_2, rois)
bar_band('RMSE set 2', folder_a + 'rmse_bands_2.png', rmses, rmses_d, rmse_stds, rmse_stds_d)
bar_band('Bias set 2', folder_a + 'bias_bands_2.png', biases, biases_d, bias_stds, bias_stds_d)
bar_band('STD set 2', folder_a + 'std_bands_2.png', stds, stds_d, std_stds, std_stds_d)

# band_std_vector_d, band_std_avg_d, band_std_std_d = band_std
# low_std_vector_d, low_std_avg_d, low_std_std_d = low_std
# print(f"band_std_d_roi: {band_std_avg_d}, {band_std_std_d}")
# print(f"low_std_d_roi: {low_std_avg_d}, {low_std_std_d}")

# rmse_f_d, bias_f_d, std_f_d = error_freq_roi(folder_a_d, nums_a, image_sets_a_d, rois)
# all_errors_f_d, error_vectors_f_d = calculate_error(rmse_f_d, bias_f_d, std_f_d, frequency=True)
# record_errors(all_errors_f_d, record_a, frequency=True, perturbation=True)
# torch.save(error_vectors_f_d, folder_a_d + 'error_vectors_d_f_roi.pt')
# torch.save(all_errors_f_d, folder_a_d + 'all_errors_d_f_roi.pt')