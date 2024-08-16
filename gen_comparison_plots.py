"""
generate the comparison plots of errors between the true image group and the true image with digits group
generate bar plots and violin plots 
"""
from eval_fn import *
from sample_fn import *
from perturbation_fn import *

folder_a = "test/"
folder_a_d = "test_d/"

image_sets_a = torch.load(folder_a + "image_sets_a.pt")
image_sets_a_d = torch.load(folder_a_d + "image_sets_a_d.pt")

all_errors = torch.load(folder_a + "all_errors.pt")
all_errors_f = torch.load(folder_a + "all_errors_f.pt")
all_errors_d = torch.load(folder_a_d + "all_errors_d.pt")
all_errors_d_f = torch.load(folder_a_d + "all_errors_d_f.pt")

error_vectors = torch.load(folder_a + 'error_vectors.pt')
error_vectors_f = torch.load(folder_a + 'error_vectors_f.pt')
error_vectors_d = torch.load(folder_a_d + 'error_vectors_d.pt')
error_vectors_d_f = torch.load(folder_a_d + 'all_errors_d_f.pt')

# plot for comparison
display_perturbation(folder_a_d + "difference.png", image_sets_a, image_sets_a_d)
# image domain
plot_error(folder_a_d, all_errors, all_errors_d, frequency=False)
# frequency domain
plot_error(folder_a_d, all_errors_f, all_errors_d_f, frequency=True)