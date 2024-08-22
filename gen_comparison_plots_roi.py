"""
generate the comparison plots of errors between the true image group and the true image with digits group
generate bar plots and violin plots 
"""
from eval_fn import *
from sample_fn import *
from perturbation_fn import *
from map_fn_roi import *

folder_a = "samples_a/"
folder_a_d = "samples_a_d/"

image_sets_a = torch.load(folder_a + "image_sets_a.pt")
image_sets_a_d = torch.load(folder_a_d + "image_sets_a_d.pt")

all_errors = torch.load(folder_a + "all_errors_roi.pt")
all_errors_f = torch.load(folder_a + "all_errors_f_roi.pt")
all_errors_d = torch.load(folder_a_d + "all_errors_d_roi.pt")
all_errors_d_f = torch.load(folder_a_d + "all_errors_d_f_roi.pt")

error_vectors = torch.load(folder_a + 'error_vectors_roi.pt')
error_vectors_f = torch.load(folder_a + 'error_vectors_f_roi.pt')
error_vectors_d = torch.load(folder_a_d + 'error_vectors_d_roi.pt')
error_vectors_d_f = torch.load(folder_a_d + 'all_errors_d_f_roi.pt')

# image domain
plot_error_roi(folder_a_d, all_errors, all_errors_d, frequency=False)
# frequency domain
plot_error_roi(folder_a_d, all_errors_f, all_errors_d_f, frequency=True)