"""
sample and generate image sets (true_images, reconstructions, measurements)
"""
from eval_fn import *
from sample_fn import *

noise_hu = 100
contrast = 200


# folder_a = "samples_a_32/"
# nums_a = 16, 1, 32, 512, 32
# image_sets_a = sample_fn(nums_a, noise_hu, contrast, perturbation=False)
# torch.save(image_sets_a, folder_a + 'image_sets_a.pt')
# meas_var_a = check_measurement_var(nums_a, image_sets_a)
# display_image_sets(folder_a, image_sets_a)
# create_animation(folder_a, "animation_a.mp4", nums_a, image_sets_a)

# folder_a_d = "samples_a_d_32/"
folder_a_d = "classifier_100/"
nums_a = 100, 1, 16, 512, 32
# with perturbation
image_sets_a_d, rois, ground_truth_label = sample_fn(nums_a, noise_hu, contrast, perturbation=True)
torch.save(image_sets_a_d, folder_a_d + 'image_sets_a_d.pt')
torch.save(rois, folder_a_d + "rois.pt")
torch.save(ground_truth_label, folder_a_d + "ground_truth_label.pt")
# meas_var_a_d = check_measurement_var(nums_a, image_sets_a_d)
# display_image_sets(folder_a_d, image_sets_a_d)
# create_animation(folder_a_d, "animation_a_d.mp4", nums_a, image_sets_a_d)

