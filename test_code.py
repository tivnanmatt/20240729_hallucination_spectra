"""
test code
"""
from eval_fn import *
from sample_fn import *

### test bar_plot_error
# bar_plot_error("Test Bar Plot", "test/test_bar", 8.2345, 1.0, 7, 3.0084)

### test digit insertion inside the sampling function
# nums = 1, 1, 1, 512, 4
# noise_hu = 100
# contrast = 10
# image_sets = sample_fn(nums, noise_hu, contrast, perturbation=False)
# true_normalized, measurement, reconstruction = image_sets
# image_sets_d, rois = sample_fn(nums, noise_hu, contrast, perturbation=True)
# perturbed_true, measurement, reconstruction = image_sets_d
# np.save('rois.npy', rois)
# display(0, "test/digit_insertion", true_normalized, perturbed_true)

### test error output
# all_errors = [12], [0.2], [10], [0.3456], [8], [0.191]
# filename = "record_error.txt"

# record_errors(all_errors, filename, freqeuncy=True, perturbation=True)

### test violin plots
# # run a small sample, print these vectors, see their shapes, 
# # see how to concatenate them in the violin plot function
# error_vectors = torch.load('test/error_vectors.pt')
# error_vectors_f = torch.load('test/error_vectors_f.pt')
# error_vectors_d = torch.load('test_d/error_vectors_d.pt')
# error_vectors_d_f = torch.load('test_d/error_vectors_d_f.pt')

# rmse, bias, std = error_vectors
# print(rmse.size())
# # tensor size [4, 1, 1, 1] -> numpy vector [1, 4] (shape: (4, ), 4 is number of patients)
# rmse = torch.squeeze(rmse).numpy()
# bias = torch.squeeze(bias).numpy()
# std = torch.squeeze(std).numpy()

# data = np.stack((rmse, bias, std), axis=0)
# print(rmse)
# print(bias)
# print(std)
# print(data)

# print("rmse vector\n", rmse)
# print("rmse vector shape\n", rmse.shape)

# violin_plot_error(error_vectors, error_vectors_f, error_vectors_d, error_vectors_d_f)

### test filter for liver identification
