import torch
import torchvision
import torchvision.transforms as T
# import tensorflow as tf
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import matplotlib.pyplot as plt
import numpy as np

def save_img(error, filename, rois, plot_min, plot_max):
    sig = 487.3876
    mu = -572.3446
    # normalize
    error = error * sig + mu
    for i in range(16, 32):
        iRow, iCol = rois[i]
        img = plt.imshow(error[i, 0, 0, 0, iRow:(iRow+28), iCol:(iCol+28)].detach().cpu(), cmap='gray', vmin=plot_min, vmax=plot_max)
        plt.axis('off')
        plt.savefig(filename + f"_{i+1}.png", bbox_inches='tight', pad_inches = 0)
        
    print("images saved")

    return 0

# Load the model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# image_sets = torch.load("samples_a_d_200/image_sets_a_d.pt")
# rois = torch.load("samples_a_d_200/rois.pt")
# true_images, measurements, reconstructions = image_sets
# save_img(reconstructions, "recon_200/perturbed_recon", rois, -160, 240)
# save_img(true_images, "recon_200/true", rois, -160, 240)
# Save images from the sampling (gen_image_sets.py)
image_sets = torch.load("samples_200_clip/image_sets_a_d.pt")
rois = torch.load("samples_200_clip/rois.pt")
true_images, measurements, reconstructions = image_sets
# labels = torch.load("samples_200_clip/ground_truth_label.pt")
save_img(reconstructions, "samples_200_clip/perturbed_recon", rois, -160, 240)
save_img(true_images, "samples_200_clip/true", rois, -160, 240)

# print("TRUE")
# for i in range(16):
# Load the image 
image = Image.open(f"samples_200_clip/true_26.png")
# image = Image.open(f"recon_200/true_17.png")

# Preprocess the image
# 1. resize from 28x28 to 224x224
# 2. convert to RGB by repeating the single channel three times, rescale to 0-255 and 8-bit uint
# 3. convert to numpy array
# 4. convert to PIL
size_transform = T.Resize(224)
img_resized = size_transform(image)
img_RGB = img_resized.convert("RGB")
tensor_trans = T.Compose([T.ToTensor()])
img_RGB = tensor_trans(img_RGB)
img_RGB = img_RGB.numpy()
img_RGB = np.moveaxis((img_RGB*255).astype(np.uint8), 0, -1)
img_PIL= Image.fromarray(img_RGB, "RGB")

# Feed the PIL image to the CLIP model with 10 options for captions
# Options
digit_options = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# Prepare the inputs
inputs = processor(text=digit_options, images=img_PIL, 
                return_tensors="pt", padding=True)

# Forward pass to get logits
outputs = model(**inputs)

# this is the image-text similarity score
logits_per_image = outputs.logits_per_image

# convert to probabilities
probs = logits_per_image.softmax(dim=1)

# Display the results
for i, diagnosis in enumerate(digit_options):
    print(f"{diagnosis}: {probs[0][i].item() * 100:.2f}%")

# Find the predicted diagnosis
predicted_idx = probs.argmax(dim=1).item()
predicted_diagnosis = digit_options[predicted_idx]

# print(f"\n{i+1} Predicted Digits: {predicted_diagnosis}")
print(f"\nPredicted Digits: {predicted_diagnosis}")






