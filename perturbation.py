from perturbation_fn import *

train_num_files = 1  #16
test_num_files = 1

# insert digits into true images
contrast = 10     #HU
root = "TCGA_LIHC"
# train true or false
true_images = load_data(root, train_num_files, test_num_files, train=True)
true_normalized = normalization(true_images)
perturbed_true = perturbation(true_images, contrast)

# display the samples
num = 50
filename = 'MNIST_sample/perturbed_50_contrast_10.png'
display(num, filename, true_normalized, perturbed_true)