from perturbation_fn import *

train_num_files = 1  #16
test_num_files = 1

# insert digits into true images
contrast = 10     #HU
root = "TCGA_LIHC"
true_images = load_data(root, train_num_files, test_num_files, train=True)
perturbed_true = perturbation(true_images, contrast)

# display the samples
display(perturbed_true)