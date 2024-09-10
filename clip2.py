"""
use clip_observer.py
"""
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from clip_observer import CLIPModelObserver
import matplotlib.pyplot as plt

def save_img(error, filename, rois, plot_min, plot_max):
    sig = 487.3876
    mu = -572.3446
    # normalize
    error = error * sig + mu
    for i in range(96):
        iRow, iCol = rois[i]
        img = plt.imshow(error[i, 0, 0, 0, iRow:(iRow+28), iCol:(iCol+28)].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
        plt.axis('off')
        plt.savefig(filename + f"_{i+1}.png", bbox_inches='tight', pad_inches = 0)
        
    print("images saved")

    return 0

# Save images from the sampling (gen_image_sets.py)
image_sets = torch.load("samples_200_clip/image_sets_a_d.pt")
rois = torch.load("samples_200_clip/rois.pt")
labels = torch.load("samples_200_clip/ground_truth_label.pt")
true_images, measurements, reconstructions = image_sets
save_img(reconstructions, "samples_200_clip/perturbed_recon", rois, -160, 240)
save_img(true_images, "samples_200_clip/true", rois, -160, 240)

observer = CLIPModelObserver(verbose=True, batch_size=16)

# Load the images
labels = labels[:96]
images = []
for i in range(100):
    image = Image.open(f"recon_200/perturbed_recon_{i+1}.png")

    # Preprocess the image
    size_transform = T.Resize(224)
    img_resized = size_transform(image)
    img_RGB = img_resized.convert("RGB")
    tensor_trans = T.Compose([T.ToTensor()])
    img_RGB = tensor_trans(img_RGB)
    img_RGB = img_RGB.numpy()
    img_RGB = np.moveaxis((img_RGB*255).astype(np.uint8), 0, -1)
    img_PIL= Image.fromarray(img_RGB, "RGB")

    images.append(img_PIL)

# Classify and evaluate results
results, predictions = observer.evaluate(images, labels)
observer.save_results(results, predictions, "clip_200.csv")
observer.print_evaluation(results, filename="clip_200.csv", predictions=predictions)