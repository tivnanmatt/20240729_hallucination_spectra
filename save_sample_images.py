"""
script to save true and reconstruction images, for the purpose of classification
"""
import torch
import matplotlib.pyplot as plt

digit_pixels = 28
plot_min = -160
plot_max = 240

def output_true(true, folder, plot_min, plot_max, rois):
    sig = 487.3876
    mu = -572.3446
    # normalize
    true = true * sig + mu
    for num in range(100):
        iRow, iCol = rois[num]
        im = plt.imshow(true[num, 0, 0, 0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
        plt.axis('off')
        plt.savefig(folder + f"true_{num+1}.png", bbox_inches='tight', pad_inches = 0)

    return 0

def output_measurement(measurements, folder, plot_min, plot_max, rois):
    sig = 487.3876
    mu = -572.3446
    # normalize
    measurements = measurements * sig + mu
    for num in range(100):
        iRow, iCol = rois[num]
        im = plt.imshow(measurements[num, 0, 0, 0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
        plt.axis('off')
        plt.savefig(folder + f"meas_{num+1}.png", bbox_inches='tight', pad_inches = 0)

    return 0

def output_recon(recon, folder, plot_min, plot_max, rois):
    sig = 487.3876
    mu = -572.3446
    # normalize
    recon = recon * sig + mu
    for num in range(100):
        iRow, iCol = rois[num]
        for r in range(16):
            im = plt.imshow(recon[num, 0, r, 0, iRow:(iRow+digit_pixels), iCol:(iCol+digit_pixels)].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
            plt.axis('off')
            plt.savefig(folder + f"recon_{num+1}-{r+1}.png", bbox_inches='tight', pad_inches = 0)
        
        if (num+1)%16==0:
            print(f"Image {num+1} is saved.")

    return 0

image_sets = torch.load("classifier_100/image_sets_a_d.pt")
rois = torch.load("classifier_100/rois.pt")
labels = torch.load("classifier_100/ground_truth_label.pt")
true_images, measurements, reconstructions = image_sets
folder = "classifier_100/measurements/"

# output_true(true_images, folder, plot_min, plot_max, rois)
output_measurement(measurements, folder, plot_min, plot_max, rois)
# output_recon(reconstructions, folder, plot_min, plot_max, rois)