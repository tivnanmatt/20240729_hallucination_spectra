"""
evaluation for different scenarios
"""
from eval_fn import *

"""
a. reconstruction given true images
N=16, M=16, R=1

"""
nums_a = 64, 16, 16, 1
image_sets_a = setup_sampling(nums_a)
meas_var_a = check_measurement_var(nums_a, image_sets_a)
display_image_sets(image_sets_a)
# error_maps(nums_a, image_sets_a)


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
