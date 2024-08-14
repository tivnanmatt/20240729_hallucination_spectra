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
    # transform to tensor will add one dimension to the 1x28x28 images
    trans = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(root='./MNIST', train=True, download=True, transform=trans)
    data_loader = torch.utils.data.DataLoader(mnist)
    digit_list = []

    for digit, label in data_loader:
        digit_list.append(digit)
    
    # convert from array to tensor
    digits = torch.cat(digit_list)
    
    # Convert to float
    digits = digits.float()

    return digits

# convert true images from HU to standard units
def normalization(true_images):
    true_normalized = (true_images - mu) / sigma
    return true_normalized

# insert digits into true images
def insert_digits(true_normalized, digits, contrast):
    num_images = true_normalized.size(dim=0)

    # go through each image in true_normalized
    for n in range(num_images):
        # randomly select a pair of row and column
        iRow = random.randrange(150, num_pixels - digit_pixels - 150)
        iCol = random.randrange(150, num_pixels - digit_pixels - 150)
        if n == 100:
            print(iRow, iCol)

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

# add digits to a single true image
def add_digits(true_image, digits, iImage, contrast):
    threshold = normalization(-300)
    true_normalized = normalization(true_image)
    inside = False

    # check if the random spot is inside patient tissue
    while(not inside):
        # randomly select a pair of row and column
        iRow = random.randrange(num_pixels - digit_pixels)
        iCol = random.randrange(num_pixels - digit_pixels)
        # check inside the tissue
        region = true_normalized[iImage, 0, 0, 0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)] 
        inside = inside_tissue(region, threshold)

    # insert one digit at the pair of row and column
    true_normalized[iImage, 0, 0, 0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)] += contrast * digits[iImage, 0, 0, 0, :, :]

    return true_normalized

# check if the region is inside tissue
def inside_tissue(region, threshold):
    # region + digit_pixels
    avg = torch.mean(region, dim=(4, 5))
    if avg >= threshold:
        return True
    else:
        return False

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


# display the differences
def display(num, filename, true_normalized, perturbed_true):
    vmin = 0
    vmax = 1
    true_display = rescale_abdomen_window(standard_to_hu(true_normalized))
    perturbed_true_display = rescale_abdomen_window(standard_to_hu(perturbed_true))
    difference = perturbed_true_display - true_display

    # plot the original true image, perturbed true image, and the difference between them
    fig, ax = plt.subplots(1, 4, figsize=(15, 5))
    im1 = ax[0].imshow(true_display[num, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
    ax[0].set_title("True Image")
    im2 = ax[1].imshow(perturbed_true_display[num, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
    ax[1].set_title("True Image with a Digit")
    im3 = ax[2].imshow(difference[num, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
    ax[2].set_title("Difference")
    im4 = ax[3].imshow(difference[num, 0, :, :].detach().cpu(), cmap='gray')
    ax[3].set_title("Difference without limits")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    plt.savefig(filename)

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