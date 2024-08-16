"""
evaluation for different scenarios
nums = num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps
"""
from eval_fn import *
from sample_fn import *

noise_hu = 20
contrast = 10

"""
a. true images without digits
N=16, M=16, R=1

"""

folder_a = "samples_a/"
nums_a = 16, 16, 16, 512, 32
record_a = 'record_error_a.txt'
# without perturbation
image_sets_a = sample_fn(nums_a, noise_hu, contrast, perturbation=False)
# save image sets, can be used for creating animation later
torch.save(image_sets_a, folder_a + 'image_sets_a.pt')
meas_var_a = check_measurement_var(nums_a, image_sets_a)
display_image_sets(folder_a, image_sets_a)
# create_animation(folder_a, "animation_a.mp4", nums_a, image_sets_a)

rmse, bias, std = error_maps(folder_a, nums_a, image_sets_a)
all_errors, error_vectors = calculate_error(rmse, bias, std, frequency=False)
record_errors(all_errors, record_a, frequency=False, perturbation=False)
torch.save(error_vectors, folder_a + 'error_vectors.pt')
torch.save(all_errors, folder_a + 'all_errors.pt')

rmse_f, bias_f, std_f = error_freq(folder_a, nums_a, image_sets_a)
all_errors_f, error_vectors_f = calculate_error(rmse_f, bias_f, std_f, frequency=True)
record_errors(all_errors_f, record_a, frequency=True, perturbation=False)
torch.save(error_vectors_f, folder_a + 'error_vectors_f.pt')
torch.save(all_errors_f, folder_a + 'all_errors_f.pt')

folder_a_d = "samples_a_d/"
# with perturbation
image_sets_a_d = sample_fn(nums_a, noise_hu, contrast, perturbation=True)
torch.save(image_sets_a, folder_a_d + 'image_sets_a_d.pt')
meas_var_a_d = check_measurement_var(nums_a, image_sets_a_d)
display_image_sets(folder_a_d, image_sets_a_d)
# create_animation(folder_a_d, "animation_a_d.mp4", nums_a, image_sets_a_d)

rmse_d, bias_d, std_d = error_maps(folder_a_d, nums_a, image_sets_a_d)
all_errors_d, error_vectors_d = calculate_error(rmse_d, bias_d, std_d, frequency=False)
record_errors(all_errors_d, record_a, frequency=False, perturbation=True)
torch.save(error_vectors_d, folder_a + 'error_vectors_d.pt')
torch.save(all_errors_d, folder_a_d + 'all_errors_d.pt')

rmse_d_f, bias_d_f, std_d_f = error_freq(folder_a_d, nums_a, image_sets_a_d)
all_errors_d_f, error_vectors_d_f = calculate_error(rmse_d_f, bias_d_f, std_d_f, frequency=True)
record_errors(all_errors_d_f, record_a, frequency=True, perturbation=True)
torch.save(error_vectors_d_f, folder_a + 'error_vectors_d_f.pt')
torch.save(all_errors_d_f, folder_a + 'all_errors_d_f.pt')

# plot for comparison
display_difference(folder_a_d, image_sets_a, image_sets_a_d)
# image domain
plot_error(folder_a_d, all_errors, all_errors_d, frequency=False)
# frequency domain
plot_error(folder_a_d, all_errors_f, all_errors_d_f, frequency=True)

"""
a. reconstruction given true images
N=16, M=16, R=1

"""
"""
folder_a = "maps_a/"
nums_a = 16, 16, 1, 512, 32
image_sets_a = sample_fn(nums_a, noise_hu)
meas_var_a = check_measurement_var(nums_a, image_sets_a)
display_image_sets(folder_a, image_sets_a)
create_animation(folder_a, "animation_a.mp4", nums_a, image_sets_a)
error_maps(folder_a, nums_a, image_sets_a)
error_freq(folder_a, nums_a, image_sets_a)
"""

"""
b. reconstruction given measurements
N=16, M=64, R=1

"""
"""
folder_b = "maps_b/"
nums_b = 16, 64, 1, 512, 32
image_sets_b = sample_fn(nums_b, noise_hu)
meas_var_b = check_measurement_var(nums_b, image_sets_b)
display_image_sets(folder_b, image_sets_b)
create_animation(folder_b, "animation_b.mp4", nums_b, image_sets_b)
error_maps(folder_b, nums_b, image_sets_b)
error_freq(folder_b, nums_b, image_sets_b)
"""

"""
c. reconstruction given measurements
N=16, M=1, R=16

"""
"""
folder_c = "maps_c/"
nums_c = 16, 1, 16, 512, 32
image_sets_c = sample_fn(nums_c, noise_hu)
meas_var_c = check_measurement_var(nums_c, image_sets_c)
display_image_sets(folder_c, image_sets_c)
create_animation(folder_c, "animation_c.mp4", nums_c, image_sets_c)
error_maps(folder_c, nums_c, image_sets_c)
error_freq(folder_c, nums_c, image_sets_c)
"""

"""
d. reconstruction given measurements
N=16, M=1, R=64

"""
"""
folder_d = "maps_d/"
nums_d = 16, 1, 64, 512, 32
image_sets_d = sample_fn(nums_d, noise_hu)
meas_var_d = check_measurement_var(nums_d, image_sets_d)
display_image_sets(folder_d, image_sets_d)
create_animation(folder_d, "animation_d.mp4", nums_d, image_sets_d)
error_maps(folder_d, nums_d, image_sets_d)
error_freq(folder_d, nums_d, image_sets_d)
"""

"""
debug
"""
"""
folder_debug = "debug/"
nums_debug = 4, 16, 1, 512, 32
image_sets_debug = sample_fn(nums_debug, noise_hu)
meas_var_debug = check_measurement_var(nums_debug, image_sets_debug)
display_image_sets(folder_debug, image_sets_debug)
# create_animation(folder_debug, "animation_debug.mp4", nums_debug, image_sets_debug)
error_maps(folder_debug, nums_debug, image_sets_debug)
# error_freq(folder_debug, nums_debug, image_sets_debug)
"""