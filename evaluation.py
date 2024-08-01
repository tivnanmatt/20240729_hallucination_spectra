"""
evaluation for different scenarios
nums = num_images, num_measurements, num_reconstructions, num_pixels, num_timesteps
"""
from eval_fn import *
from eval import *

noise_hu = 20

"""
a. reconstruction given true images
N=16, M=16, R=1

"""
folder_a = "maps_a/"
nums_a = 4, 1, 1, 512, 32
image_sets_a = sample_recon(nums_a, noise_hu)
meas_var_a = check_measurement_var(nums_a, image_sets_a)
display_image_sets(folder_a, image_sets_a)
create_animation(folder_a, "animation_a.mp4", nums_a, image_sets_a)
error_maps(folder_a, nums_a, image_sets_a)
error_freq(folder_a, nums_a, image_sets_a)


"""
b. reconstruction given measurements
N=16, M=1, R=16

"""
"""
nums_b = 64, 16, 1, 16
true_images_b, measurements_b, reconstructions_b = setup_sampling(nums_b)
meas_var_b = check_measurement_var(nums_b, image_sets_b)
display_image_sets(image_sets_b)
# error_maps(nums_b, image_sets_b)
"""
