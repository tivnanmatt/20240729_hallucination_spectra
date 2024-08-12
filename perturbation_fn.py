import torch
import torchvision.datasets as datasets
from torchvision import transforms
import random
import matplotlib.pyplot as plt

sigma = 487.3876
mu = -572.3446
num_pixels = 512
digit_pixels = 28

# function to sample MNIST digits
def sample_digits():
    trans = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./MNIST', train=True, download=True, transform=trans)
    data_loader = torch.utils.data.DataLoader(mnist)
    digit_list = []
    for digit, label in data_loader:
        digit_list.append(digit)
    digits = torch.cat(digit_list)
    
    # similar processing with the true images
    # Add an extra dimension for channels on axis 1
    # digits = torch.unsqueeze(digits, 1)
    # Convert to float
    digits = digits.float()

    return digits

# function to convert true images from HU to standard units
def normalization(true_images):
    true_normalized = (true_images - mu) / sigma
    return true_normalized

# function to insert digits into true images
def insert_digits(true_normalized, digits, contrast):
    num_images = true_normalized.size(dim=0)
    # go through each image in true_normalized
    for n in range(num_images):
        # randomly select a pair of row and column
        iRow = random.randrange(num_pixels - digit_pixels)
        iCol = random.randrange(num_pixels - digit_pixels)
        # insert one digit at the pair of row and column
        true_normalized[n, 0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)] += contrast * digits[n, 0, :, :]

    return true_normalized

# create perturbed true images
def perturbation(true_images, contrast):
    contrast_normalized = normalization(contrast)
    digits = sample_digits()
    true_normalized = normalization(true_images)
    perturbed_true = insert_digits(true_normalized, digits, contrast_normalized)

    return perturbed_true

# load data, modified from the code in laboratory
def load_data(root, train_num_files, test_num_files, train):
    if train:    
        image_list = []
        for i in range(train_num_files):
            print(f'Loading {root}/training/training_TCGA_LIHC_{str(i).zfill(6)}.pt')
            image_list.append(torch.load(root + f'/training/training_TCGA_LIHC_' + str(i).zfill(6) + '.pt'))
        images = torch.cat(image_list)

    else:
        image_list = []
        for i in range(test_num_files):
            print(f'Loading {root}/testing/testingTCGA_LIHC_{str(i).zfill(6)}.pt')
            image_list.append(torch.load(root + f'/testing/testing_TCGA_LIHC_' + str(i).zfill(6) + '.pt'))
        images = torch.cat(image_list)

    # Add an extra dimension for channels on axis 1
    images = torch.unsqueeze(images, 1)

    # Convert to float
    images = images.float()

    return images

def display(perturbed_true):
    perturbed_true_display = rescale_abdomen_window(standard_to_hu(perturbed_true))
    fig = plt.figure(figsize=(5, 5))
    vmin = 0
    vmax = 1
    plt.imshow(perturbed_true_display[0, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title('True Image with a Digit')
    plt.savefig('MNIST_sample/perturbed.png')

    return 0

# Convert standard units to Hounsfield Units (HU)
def standard_to_hu(tensor, mu=-572.3447, sigma=487.3876):
    return tensor * sigma + mu

# Rescale to (0, 1) using abdomen window and clip values outside the range
def rescale_abdomen_window(tensor, window_center=40, window_width=400):
    min_hu = window_center - (window_width / 2)
    max_hu = window_center + (window_width / 2)
    tensor = (tensor - min_hu) / (max_hu - min_hu)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor