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
"""
folder_a = "maps_a/"
nums_a = 16, 16, 1, 512, 32
image_sets_a = sample_recon(nums_a, noise_hu)
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
image_sets_b = sample_recon(nums_b, noise_hu)
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
folder_c = "maps_c/"
nums_c = 16, 1, 16, 512, 32
image_sets_c = sample_recon(nums_c, noise_hu)
meas_var_c = check_measurement_var(nums_c, image_sets_c)
display_image_sets(folder_c, image_sets_c)
create_animation(folder_c, "animation_c.mp4", nums_c, image_sets_c)
error_maps(folder_c, nums_c, image_sets_c)
error_freq(folder_c, nums_c, image_sets_c)
