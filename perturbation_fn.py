import torch
import torchvision.datasets as datasets
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import numpy as np


sigma = 487.3876
mu = -572.3446
num_pixels = 512
digit_pixels = 28


# function to sample MNIST digits
def sample_digits():
   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   # transform to tensor will add one dimension to the 1x28x28 images
   trans = transforms.Compose([transforms.ToTensor()])
   mnist = datasets.MNIST(root='./MNIST', train=True, download=True, transform=trans)
   data_loader = torch.utils.data.DataLoader(mnist)
   digit_list = []


   for digit, label in data_loader:
       digit_list.append(digit)
  
   # convert from array to tensor
   digits = torch.cat(digit_list)
  
   # Convert to GPU
   # digits = digits.float().to(device)




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
   # contrast_normalized = normalization(contrast)
   contrast_normalized = contrast / sigma
   # threshold = -300 - sigma
   threshold = 100 / sigma
   inside = False
   perturbed_image = true_image.detach().clone()


   # check if the random spot is inside patient tissue
   while(not inside):
       # randomly select a pair of row and column
       iRow = random.randrange(num_pixels - digit_pixels)
       iCol = random.randrange(num_pixels - digit_pixels)
       # check inside the tissue
       region = perturbed_image[0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)]
       # inside = inside_tissue(region, threshold)
       inside = inside_liver(region, threshold)


   # save the iRow and iCol
   pos = (iRow, iCol)
   print(pos)
   # insert one digit at the pair of row and column
   digit = digits[iImage]
   perturbed_image[0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)] = perturbed_image[0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)] + contrast_normalized * digit[0, :, :]
   return perturbed_image, pos


# check if the region is inside tissue
def inside_tissue(region, threshold):
   print(threshold)
   avg = torch.mean(region)
   if avg >= threshold:
       print(avg)
       return True
   else:
       return False
  
def inside_liver(region, threshold):
   region_blurred = gaussian_filter(region, sigma=5)
   avg = np.abs(np.mean(region_blurred - region))
   if avg < 1e-4:
       print(avg)
       return True
   else:
       return False
  
# def inside_liver(region, threshold):
#     region = gaussian_filter(region, sigma=5)
#     avg = np.mean(region)
#     if avg >= threshold:
#         print(avg)
#         return True
#     else:
#         return False
  
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
   # difference = perturbed_true_display - true_display
   difference = perturbed_true - true_normalized
   difference_display = rescale_abdomen_window(standard_to_hu(difference))


   # plot the original true image, perturbed true image, and the difference between them
   fig, ax = plt.subplots(1, 3, figsize=(15, 5))
   # im1 = ax[0].imshow(true_display[num, 0, 0, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
   # ax[0].set_title("True Image")
   # im2 = ax[1].imshow(perturbed_true_display[num, 0, 0, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
   # ax[1].set_title("True Image with a Digit")
   # im3 = ax[2].imshow(difference_display[num, 0, 0, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
   # ax[2].set_title("Difference")
  
   im1 = ax[0].imshow(true_display[0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
   ax[0].set_title("True Image")
   im2 = ax[1].imshow(perturbed_true_display[0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
   ax[1].set_title("True Image with a Digit")
   im3 = ax[2].imshow(difference[0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
   ax[2].set_title("Difference")


   # im1 = ax[0].imshow(true_normalized[0, :, :].detach().cpu(), cmap='gray')
   # ax[0].set_title("True Image")
   # im2 = ax[1].imshow(perturbed_true[0, :, :].detach().cpu(), cmap='gray')
   # ax[1].set_title("True Image with a Digit")
   # im3 = ax[2].imshow(difference[0, :, :].detach().cpu(), cmap='gray')
   # ax[2].set_title("Difference")


   for a in ax:
       a.set_xticks([])
       a.set_yticks([])
   plt.savefig(filename)


   return 0


def display_perturbation(filename, image_sets, image_sets_d):
   true_images, measurements, reconstructions = image_sets
   true_images_d, measurements_d, reconstructions_d = image_sets_d
   true_display = rescale_abdomen_window(standard_to_hu(true_images.detach().cpu()))
   perturbed_true_display = rescale_abdomen_window(standard_to_hu(true_images_d.detach().cpu()))
   difference = perturbed_true_display - true_display
   vmin = 0
   vmax = 1
   num = 0
   fig, ax = plt.subplots(4, 4, figsize=(15, 5))
   for col in range(4):
       for row in range(4):
           im = ax[col, row].imshow(difference[num, 0, 0, 0, :, :].detach().cpu(), cmap='gray', vmin=vmin, vmax=vmax)
           ax[col, row].set_xticks([])
           ax[col, row].set_yticks([])
           num += 1
           ax[col, row].set_title(f"{num}", y=0.95, fontsize=8)
   # colorbar
   fig.subplots_adjust(left=0.0,
                           bottom=0.05,
                           right=0.3,
                           top=0.9,
                           wspace=0.0,
                           hspace=0.2)
   cbar_ax = fig.add_axes([0.31, 0.05, 0.01, 0.8])
   color_bar = fig.colorbar(im, cax=cbar_ax)
   color_bar.minorticks_on()


   fig.suptitle("Difference", x=0.15, fontsize=10)
   plt.savefig(filename, bbox_inches = 'tight', pad_inches = 0.2)
   print(filename + " saved")


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

