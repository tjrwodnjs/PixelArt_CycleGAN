import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

image_number = 900

# Save Domain A Dataset as npy
dataset_A = []

for i in range(1, image_number+1):
    img = Image.open(f'./training_dataset/trainA/{i}.png')
    if np.array(img).shape != (256, 256, 3):
        img = img.resize((256,256))
    img = np.asarray(img) / 255
    dataset_A.append(img)

dataset_A = np.array(dataset_A)
np.save(f'Domain_A_{image_number}_256x256', dataset_A)

print(dataset_A.shape)

# Save Domain B Dataset as npy
dataset_B = []

for i in range(1, image_number+1):
    img = Image.open(f'./training_dataset/trainB/{i}.png')
    if np.array(img).shape != (256, 256, 3):
        img = img.resize((256,256))
    img = np.asarray(img) / 255
    dataset_B.append(img)

dataset_B = np.array(dataset_B)
np.save(f'Domain_B_{image_number}_256x256', dataset_B)

print(dataset_B.shape)

