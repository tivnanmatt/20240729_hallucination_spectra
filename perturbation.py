from perturbation_fn import *

train_num_files = 1  #16
test_num_files = 1

# insert digits into true images
contrast = 10    #HU
root = "TCGA_LIHC"
# train true or false
true_images = load_data(root, train_num_files, test_num_files, train=False)
true_normalized = normalization(true_images)
# perturbed_true = perturbation(true_images, contrast)
digits = sample_digits()
perturbed_true = add_digits(true_normalized[50], digits, 50, contrast)

# display the samples
num = 50
filename = 'MNIST_sample/perturbed.png'
display(num, filename, true_images, perturbed_true)