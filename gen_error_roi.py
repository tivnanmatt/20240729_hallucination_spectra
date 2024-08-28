"""
Plot error maps that focus on the digit-inserted regions
For both original true images and perturbed true image reconstructions
"""

from eval_fn import *
from sample_fn import *
from map_fn_roi import *
from filter_fn import *

folder_a = "samples_a/"
folder_a_d = "samples_a_d/"
record_a = 'record_a_roi.txt'
nums_a = 16, 1, 16, 512, 32
image_sets_a = torch.load("samples_a/image_sets_a.pt")
image_sets_a_d = torch.load("samples_a_d/image_sets_a_d.pt")
rois = torch.load("samples_a_d/rois.pt")

display_rois(folder_a, image_sets_a, rois)

rmse, bias, std = error_maps_roi(folder_a, nums_a, image_sets_a, rois)
all_errors, error_vectors = calculate_error(rmse, bias, std, frequency=False)
record_errors(all_errors, record_a, frequency=False, perturbation=False)
torch.save(error_vectors, folder_a + 'error_vectors_roi.pt')
torch.save(all_errors, folder_a + 'all_errors_roi.pt')

rmse_f, bias_f, std_f = error_freq_roi(folder_a, nums_a, image_sets_a, rois)
all_errors_f, error_vectors_f = calculate_error(rmse_f, bias_f, std_f, frequency=True)
record_errors(all_errors_f, record_a, frequency=True, perturbation=False)
torch.save(error_vectors_f, folder_a + 'error_vectors_f_roi.pt')
torch.save(all_errors_f, folder_a + 'all_errors_f_roi.pt')

display_rois(folder_a_d, image_sets_a_d, rois)

# with digits
rmse_d, bias_d, std_d = error_maps_roi(folder_a_d, nums_a, image_sets_a_d, rois)
all_errors_d, error_vectors_d = calculate_error(rmse_d, bias_d, std_d, frequency=False)
record_errors(all_errors_d, record_a, frequency=False, perturbation=True)
torch.save(error_vectors_d, folder_a_d + 'error_vectors_d_roi.pt')
torch.save(all_errors_d, folder_a_d + 'all_errors_d_roi.pt')

rmse_f_d, bias_f_d, std_f_d = error_freq_roi(folder_a_d, nums_a, image_sets_a_d, rois)
all_errors_f_d, error_vectors_f_d = calculate_error(rmse_f_d, bias_f_d, std_f_d, frequency=True)
record_errors(all_errors_f_d, record_a, frequency=True, perturbation=True)
torch.save(error_vectors_f_d, folder_a_d + 'error_vectors_d_f_roi.pt')
torch.save(all_errors_f_d, folder_a_d + 'all_errors_d_f_roi.pt')